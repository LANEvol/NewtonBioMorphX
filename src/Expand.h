// Copyright (C) 2026 Ebrahim Jahanbakhsh & Michel Milinkovitch
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef EXPAND_H_
#define EXPAND_H_

#include <thrust/device_vector.h>
#include <thrust/gather.h>

template <typename InputIterator1,
        typename InputIterator2,
        typename OutputIterator>
OutputIterator expand(InputIterator1 first1,
                      InputIterator1 last1,
                      InputIterator2 first2,
                      OutputIterator output) {
    typedef typename thrust::iterator_difference<InputIterator1>::type difference_type;

    difference_type input_size  = thrust::distance(first1, last1);
    difference_type output_size = thrust::reduce(first1, last1);

    // scan the counts to obtain output offsets for each input element
    thrust::device_vector<difference_type> output_offsets(input_size, 0);
    thrust::exclusive_scan(first1, last1, output_offsets.begin());

    // scatter the nonzero counts into their corresponding output positions
    thrust::device_vector<difference_type> output_indices(output_size, 0);
    thrust::scatter_if
            (thrust::counting_iterator<difference_type>(0),
             thrust::counting_iterator<difference_type>(input_size),
             output_offsets.begin(),
             first1,
             output_indices.begin());

    // compute max-scan over the output indices, filling in the holes
    thrust::inclusive_scan
            (output_indices.begin(),
             output_indices.end(),
             output_indices.begin(),
             thrust::maximum<difference_type>());

    // gather input values according to index array (output = first2[output_indices])
    OutputIterator output_end = output; thrust::advance(output_end, output_size);
    thrust::gather(output_indices.begin(),
                   output_indices.end(),
                   first2,
                   output);

    // return output + output_size
    thrust::advance(output, output_size);
    return output;
}

template<typename InputIterator1,
        typename InputIterator2,
        typename OutputIterator>
void expand_and_increment(InputIterator1 first1,
                          InputIterator1 last1,
                          InputIterator2 first2,
                          OutputIterator output) {
    typedef typename thrust::iterator_difference<InputIterator1>::type difference_type;
    difference_type input_size  = thrust::distance(first1, last1);
    difference_type output_size = thrust::reduce(first1, last1);
    // scan the counts to obtain output offsets for each input element
    thrust::device_vector<difference_type> output_offsets(input_size);
    thrust::exclusive_scan(first1, last1, output_offsets.begin());
    // scatter the nonzero counts into their corresponding output positions
    thrust::device_vector<difference_type> output_indices(output_size);
    thrust::scatter_if
            (thrust::counting_iterator<difference_type>(0),
             thrust::counting_iterator<difference_type>(input_size),
             output_offsets.begin(),
             first1,
             output_indices.begin());
    // compute max-scan over the output indices, filling in the holes
    thrust::inclusive_scan
            (output_indices.begin(),
             output_indices.end(),
             output_indices.begin(),
             thrust::maximum<difference_type>());
    // gather input values according to index array (output = first2[output_indices])
    OutputIterator output_end = output; thrust::advance(output_end, output_size);
    thrust::gather(output_indices.begin(),
                   output_indices.end(),
                   first2,
                   output);
    // rank output_indices
    thrust::device_vector<difference_type> ranks(output_size);
    thrust::exclusive_scan_by_key(output_indices.begin(), output_indices.end(),
                                  thrust::make_constant_iterator<difference_type>(1),
                                  ranks.begin());
    // increment output by ranks
    thrust::transform(output, output + output_size, ranks.begin(), output, thrust::placeholders::_1 + thrust::placeholders::_2);
}

#endif /* EXPAND_H_ */