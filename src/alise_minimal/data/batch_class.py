"""Definition of class associated to some batches"""

from dataclasses import dataclass

from torch import Tensor


@dataclass
class SITSBatch:
    """
    sits: B T C H W
    positions: B T
    pad_mask: B T boolean Tensor. True means the value will be
    """

    sits: Tensor
    positions: Tensor
    pad_mask: Tensor
    cld_mask: Tensor | None = None

    def pin_memory(self):
        """
        To optimize data transfer
        Returns
        -------

        """
        self.sits = self.sits.pin_memory()
        self.positions = self.positions.pin_memory()
        self.pad_mask = self.pad_mask.pin_memory()
        if self.cld_mask is not None:
            self.cld_mask = self.cld_mask.pin_memory()
        return self

    def to_device(self, device):
        """
        To transfer data on device
        Parameters
        ----------
        device :

        Returns
        -------

        """
        self.sits = self.sits.to(device)
        self.positions = self.positions.to(device)
        self.pad_mask = self.pad_mask.to(device)
        if self.cld_mask is not None:
            self.cld_mask = self.cld_mask.to(device)
        return self


@dataclass
class SegBatch:
    """
    sits: B T C H W
    positions: B T
    pad_mask: B T boolean Tensor. True means the value will be
    *ignored* in the attention
    labels: B H W (long Tensor)
    """

    sits: Tensor
    positions: Tensor
    pad_mask: Tensor
    labels: Tensor
    labels_mask: Tensor | None = None


@dataclass
class CDBInput:
    """
    A Batch of CropRot samples useful for change detection
    """

    year1: SITSBatch
    year2: SITSBatch
    label: Tensor
    mask_label: Tensor

    def pin_memory(self):
        """
        To optimize data transfer
        Returns
        -------

        """
        self.year1 = self.year1.pin_memory()
        self.year2 = self.year2.pin_memory()
        self.mask_label = self.mask_label.pin_memory()
        self.label = self.label.pin_memory()
        return self

    def to_device(self, device):
        """
        To transfer data on device
        Parameters
        ----------
        device :

        Returns
        -------

        """
        self.year1 = self.year1.to_device(device)
        self.year2 = self.year2.to_device(device)
        self.label = self.label.to(device)
        self.mask_label = self.mask_label.to(device)
        return self
