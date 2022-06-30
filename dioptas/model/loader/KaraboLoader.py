# -*- coding: utf-8 -*-
# Dioptas - GUI program for fast processing of 2D X-ray diffraction data
# Principal author: Clemens Prescher (clemens.prescher@gmail.com)
# Copyright (C) 2014-2019 GSECARS, University of Chicago, USA
# Copyright (C) 2015-2018 Institute for Geology and Mineralogy, University of Cologne, Germany
# Copyright (C) 2019-2020 DESY, Hamburg, Germany
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from bisect import bisect_right
import numpy as np

try:
    from extra_data import H5File
    from extra_data.components import identify_multimod_detectors, MultimodDetectorBase
    from extra_geom import AGIPD_1MGeometry
except ImportError:
    karabo_installed = False
else:
    karabo_installed = True

__all__ = ['EuXFELFile', 'karabo_installed']


def _frame_to_train_pulse(frame, frame_counts):
    cumsum = np.cumsum(frame_counts)
    train = bisect_right(cumsum, frame)
    pulse = int(frame - cumsum[train-1]) if train > 0 else int(frame)
    return train, pulse


class EuXFELFile:
    def __init__(self, filename):
        if not karabo_installed:
            raise IOError('karabo_data is required to load karabo h5 files')
        try:
            self.f = H5File(filename)
        except KeyError:
            raise IOError('This hdf5 file is not a generated at EuXFEL')

        self.sources = self._find_sources(self.f)
        self.select_source(next(s for s in self.sources))

    def get_image(self, ind):
        if isinstance(self.current_source, MultimodDetectorBase):
            train, frame = _frame_to_train_pulse(ind, self.current_source.frame_counts.values)
            data = self.current_source.select_trains(train).get_array('image.data', pulses=frame)
            # TODO assemble data
            return data.squeeze()
        else:
            return self.current_source[ind].ndarray().squeeze()

    def select_source(self, source):
        self.current_source = self.sources[source]

        if isinstance(self.current_source, MultimodDetectorBase):
            self.series_max = self.current_source.frame_counts.sum()
        else:
            self.series_max = self.current_source.data_counts().sum()

    def _find_sources(self, h5_file):
        sources = {}
        for name, cls in identify_multimod_detectors(h5_file):
            sources[name] = cls(h5_file, name)

        # single module detectors
        for source in h5_file.instrument_sources:
            if any(source.startswith(s) for s in sources):
                continue  # some detectors are listed in instrument_sources

            source_data = h5_file[source]
            if 'data.image.pixels' in source_data.keys():
                sources[source] = source_data['data.image.pixels']
            elif 'image.data' in source_data.keys():
                sources[source] = source_data['image.data']
            elif 'data.adc' in source_data.keys():
                sources[source] = source_data['data.adc']

        return sources
