"""SDSS benchmark loader from wwdc_spectra.
"""
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from spender.instrument import get_skyline_mask


class SDSS(Dataset):
    """Load archived SDSS splits with the original spectral preprocessing.

    ``data_root`` contains ``sdss/sdss_II_catalog``. This local copy supports
    the benchmark's catalogue conditioning (Z), not NORM conditioning, and
    never creates or downloads dataset splits. Set normalise=False to retain
    observed flux amplitudes; grid placement and nonfinite cleanup are unchanged.
    """
    def __init__(
        self,
        data_root,
        split="train", 
        x_ds=None, 
        y_catalog=None, 
        return_id=False, 
        return_pos=False,
        return_mask_ratio=False,
        return_mask=False,
        normalise=True,
    ):
        data_dir = Path(data_root) / "sdss/sdss_II_catalog"
        if not (data_dir / "dataset_dict.json").is_file():
            raise FileNotFoundError(f"Existing SDSS split dataset required: {data_dir}")

        if split not in ["train", "test", "val"]:
            msg = f"Invalid split: {split}. \
            Must be 'train' or 'test' or 'val'."
            raise ValueError(msg)

        self.normalise = normalise
        self.x_ds = x_ds
        self.y_catalog = y_catalog

        self.dataset = load_from_disk(str(data_dir))[split]
        self._wave_obs = 10 ** torch.arange(3.578, 3.97, 0.0001)
        self._skyline_mask = get_skyline_mask(self._wave_obs)      
        self.return_id = return_id  
        self.return_pos = return_pos
        self.return_mask_ratio = return_mask_ratio
        self.return_mask = return_mask
        active_variables = set(self.y_catalog["variables"]) - set(self.y_catalog["drop_variables"]) if self.y_catalog else set()
        if "NORM" in active_variables:
            raise ValueError("This benchmark loader does not support NORM conditioning")

    def prepare_spectrum(
        self,
        flux,
        ivar,
        wavelengths,
        mask,
        z,
    ):
        """Prepare spectrum for analysis

        This method creates an extended mask, using the original SDSS `and_mask` and
        the skyline mask of the instrument. The weights for all masked regions is set to
        0, but the spectrum itself is not altered.

        The spectrum and weights are then cast into a fixed-format data vector, which
        standardizes the variable wavelengths of each observation.

        A normalization is computed as the median flux in the relatively flat region
        between restframe 5300 and 5850 A. The spectrum is divided by this factor, the weights
        are muliplied with the square of this factor. With normalise=False,
        this division is skipped and norm is returned as 1.

        Parameter
        ---------
        flux: float
            Spectra flux
        ivar: float
            Invariance variance of spectrum
        wavelengths: float
            Wavelengths for each flux measurement
        mask: bool
            Masks for flux measurements
        z: float
            Redshift of the spectrum
        z_err: float
            Uncertainity in redshift.


        Returns
        -------
        spec: `torch.tensor`, shape (L, )
            Normalized spectrum
        w: `torch.tensor`, shape (L, )
            Inverse variance weights of normalized spectrum
        norm: `torch.tensor`, shape (1, )
            Flux normalization factor
        z: `torch.tensor`, shape (1, )
            Redshift (only returned when argument z=None)
        zerr: `torch.tensor`, shape (1, )
            Redshift error (only returned when argument z=None)
        """
        
        loglam = torch.log10(
            torch.tensor(wavelengths)
        )
        flux = torch.tensor(flux, dtype=torch.float32)  #np.array(obj["spectrum"]["flux"])

        flux_nan = flux.isnan()

        ivar = torch.tensor(ivar, dtype=torch.float32) #np.array(obj["spectrum"]["ivar"])
        ivar[mask] = 0

        # loglam is subset of _wave_obs, need to insert into extended tensor
        L = len(self._wave_obs)
        start = int(
            torch.round((loglam[0] - torch.log10(self._wave_obs[0]).item()) / 0.0001)
        )
        if start < 0:
            flux = flux[-start:]
            ivar = ivar[-start:]
            end = min(start + len(loglam), L)
            start = 0
        else:
            end = min(start + len(loglam), L)
        spec = torch.zeros(L)
        w = torch.zeros(L)

        spec[start:end] = flux
        w[start:end] = ivar

        # remove regions around skylines
        w[self._skyline_mask] = 0

        # normalize spectrum:
        # for redshift invariant encoder: select norm window in restframe
        wave_rest = self._wave_obs / (1 + z)
        norm = torch.tensor(1.0)
        if self.normalise:
            # flatish region that is well observed out to z ~ 0.5
            sel = (w > 0) & (wave_rest > 5300) & (wave_rest < 5850)
            if sel.count_nonzero() == 0:
                norm = torch.tensor(0)
            else:
                norm = torch.median(spec[sel])
            # remove spectra (from training) for which no valid norm could be found
            if not torch.isfinite(norm):
                norm = 0
            else:
                spec /= norm
            w *= norm**2

        # This is required to remove NaNs from the spectrum, if present.
        # Otherwise, the loss is poisioned.
        finite_spec = torch.isfinite(spec)
        finite_w = torch.isfinite(w)
        good = finite_spec & finite_w
        # zero spectrum where non-finite; zero weights where either is bad
        spec = torch.where(finite_spec, spec, torch.zeros_like(spec))
        w = torch.where(good, w, torch.zeros_like(w))

        return spec, w, norm, wave_rest, z

    @staticmethod
    def _to_f32_or_nan(x):
        return torch.tensor(
            float(x) if x is not None else float("nan"), dtype=torch.float32
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # pre-process spectrum
        spec, w, norm, wave_rest, z = self.prepare_spectrum(
            flux=item["spectrum"]["flux"],
            ivar=item["spectrum"]["ivar"],
            wavelengths=item["spectrum"]["lambda"],
            mask=item["spectrum"]["mask"],
            z=item["Z"], # Changed from original
        )

        # Pre-process conditioning data
        y = torch.stack(
            [
                torch.as_tensor(item[k])
                for k in self.y_catalog["variables"]
                if k not in self.y_catalog["drop_variables"]
            ]
        )

        out = [spec, y]
        if self.return_id:
            objid = item.get("BESTOBJID")
            specid = item.get("object_id")
            out.append(specid if specid is not None else "")
            out.append(objid if objid is not None else "")
            
        if self.return_pos:
            ra = item.get("ra"); dec = item.get("dec"); z = item.get("Z")
            out.extend([
                float(ra) if ra is not None else float("nan"),
                float(dec) if dec is not None else float("nan"),
                float(z) if z is not None else float("nan"),
            ])

        if self.return_mask_ratio:       
            out.append(
                (w == 0).sum()/len(w)
            )
        
        if self.return_mask:
            out.append(w)

        return tuple(out)

