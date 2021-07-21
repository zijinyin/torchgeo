import os
from pathlib import Path

import pytest
import torch
from rasterio.crs import CRS

from torchgeo.datasets import BoundingBox, Landsat8, ZipDataset
from torchgeo.transforms import Identity


class TestLandsat8:
    @pytest.fixture
    def dataset(self) -> Landsat8:
        root = os.path.join("tests", "data")
        bands = ["B4", "B3", "B2"]
        transforms = Identity()
        return Landsat8(root, bands=bands, transforms=transforms)

    def test_getitem(self, dataset: Landsat8) -> None:
        print(dataset.bounds.intersects(dataset.bounds))
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)

    def test_add(self, dataset: Landsat8) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ZipDataset)

    def test_plot(self, dataset: Landsat8) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x["image"], query)

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No Landsat data was found in "):
            Landsat8(str(tmp_path))

    def test_invalid_query(self, dataset: Landsat8) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* is not within bounds of the index:"
        ):
            dataset[query]