// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/fast_rcnn_layers.hpp"

#ifdef _MSC_VER
#define round(x) ((int)((x) + 0.5))
#endif

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
struct PreCalc {
  int pos1;
  int pos2;
  int pos3;
  int pos4;
  Dtype w1;
  Dtype w2;
  Dtype w3;
  Dtype w4;
};

template <typename Dtype>
void pre_calc_for_bilinear_interpolate(
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int iy_upper,
    const int ix_upper,
    Dtype roi_start_h,
    Dtype roi_start_w,
    Dtype bin_size_h,
    Dtype bin_size_w,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    std::vector<PreCalc<Dtype>>& pre_calc) {
  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      for (int iy = 0; iy < iy_upper; iy++) {
        const Dtype yy = roi_start_h + ph * bin_size_h +
            static_cast<Dtype>(iy + .5f) * bin_size_h /
                static_cast<Dtype>(roi_bin_grid_h); // e.g., 0.5, 1.5
        for (int ix = 0; ix < ix_upper; ix++) {
          const Dtype xx = roi_start_w + pw * bin_size_w +
              static_cast<Dtype>(ix + .5f) * bin_size_w /
                  static_cast<Dtype>(roi_bin_grid_w);

          Dtype x = xx;
          Dtype y = yy;
          // deal with: inverse elements are out of feature map boundary
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            PreCalc<Dtype> pc;
            pc.pos1 = 0;
            pc.pos2 = 0;
            pc.pos3 = 0;
            pc.pos4 = 0;
            pc.w1 = 0;
            pc.w2 = 0;
            pc.w3 = 0;
            pc.w4 = 0;
            pre_calc[pre_calc_index] = pc;
            pre_calc_index += 1;
            continue;
          }

          if (y <= 0) {
            y = 0;
          }
          if (x <= 0) {
            x = 0;
          }

          int y_low = (int)y;
          int x_low = (int)x;
          int y_high;
          int x_high;

          if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (Dtype)y_low;
          } else {
            y_high = y_low + 1;
          }

          if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (Dtype)x_low;
          } else {
            x_high = x_low + 1;
          }

          Dtype ly = y - y_low;
          Dtype lx = x - x_low;
          Dtype hy = 1. - ly, hx = 1. - lx;
          Dtype w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

          // save weights and indeces
          PreCalc<Dtype> pc;
          pc.pos1 = y_low * width + x_low;
          pc.pos2 = y_low * width + x_high;
          pc.pos3 = y_high * width + x_low;
          pc.pos4 = y_high * width + x_high;
          pc.w1 = w1;
          pc.w2 = w2;
          pc.w3 = w3;
          pc.w4 = w4;
          pre_calc[pre_calc_index] = pc;

          pre_calc_index += 1;
        }
      }
    }
  }
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROIPoolingParameter roi_pool_param = this->layer_param_.roi_pooling_param();
  CHECK_GT(roi_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = roi_pool_param.pooled_h();
  pooled_width_ = roi_pool_param.pooled_w();
  spatial_scale_ = roi_pool_param.spatial_scale();
  sampling_ratio_ = 2;
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);

}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->num();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
   for (int n = 0; n < num_rois; n++) {
    int index_n = n * channels_ * pooled_width_ * pooled_height_;

    // roi could have 4 or 5 columns
    const Dtype* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = 0;
    roi_batch_ind = offset_bottom_rois[0];
    offset_bottom_rois++;


    // Do not using rounding; this implementation detail is critical
    Dtype roi_start_w = offset_bottom_rois[0] * spatial_scale_;
    Dtype roi_start_h = offset_bottom_rois[1] * spatial_scale_;
    Dtype roi_end_w = offset_bottom_rois[2] * spatial_scale_;
    Dtype roi_end_h = offset_bottom_rois[3] * spatial_scale_;
    // Dtype roi_start_w = round(offset_bottom_rois[0] * spatial_scale_);
    // Dtype roi_start_h = round(offset_bottom_rois[1] * spatial_scale_);
    // Dtype roi_end_w = round(offset_bottom_rois[2] * spatial_scale_);
    // Dtype roi_end_h = round(offset_bottom_rois[3] * spatial_scale_);

    // Force malformed ROIs to be 1x1
    Dtype roi_width = std::max(roi_end_w - roi_start_w, (Dtype)1.);
    Dtype roi_height = std::max(roi_end_h - roi_start_h, (Dtype)1.);
    Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height_);
    Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width_);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio_ > 0)
        ? sampling_ratio_
        : ceil(roi_height / pooled_height_); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio_ > 0) ? sampling_ratio_ : ceil(roi_width / pooled_width_);

    // We do average (integral) pooling inside a bin
    const Dtype count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    // we want to precalculate indeces and weights shared by all chanels,
    // this is the key point of optimiation
    std::vector<PreCalc<Dtype>> pre_calc(
        roi_bin_grid_h * roi_bin_grid_w * pooled_width_ * pooled_height_);
    pre_calc_for_bilinear_interpolate(
        height_,
        width_,
        pooled_height_,
        pooled_width_,
        roi_bin_grid_h,
        roi_bin_grid_w,
        roi_start_h,
        roi_start_w,
        bin_size_h,
        bin_size_w,
        roi_bin_grid_h,
        roi_bin_grid_w,
        pre_calc);

      for (int c = 0; c < channels_; c++) {
        int index_n_c = index_n + c * pooled_width_ * pooled_height_;
        const Dtype* offset_bottom_data =
            bottom_data + (roi_batch_ind * channels_ + c) * height_ * width_;
        int pre_calc_index = 0;

        for (int ph = 0; ph < pooled_height_; ph++) {
          for (int pw = 0; pw < pooled_width_; pw++) {
            int index = index_n_c + ph * pooled_width_ + pw;

            Dtype output_val = 0.;
            for (int iy = 0; iy < roi_bin_grid_h; iy++) {
              for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                PreCalc<Dtype> pc = pre_calc[pre_calc_index];
                output_val += pc.w1 * offset_bottom_data[pc.pos1] +
                    pc.w2 * offset_bottom_data[pc.pos2] +
                    pc.w3 * offset_bottom_data[pc.pos3] +
                    pc.w4 * offset_bottom_data[pc.pos4];

                pre_calc_index += 1;
              }
            }
            output_val /= count;

            top_data[index] = output_val;
          } // for pw
        } // for ph
      } // for c


  } // for n
}


template <typename Dtype>
void ROIAlignLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}


#ifdef CPU_ONLY
STUB_GPU(ROIAlignLayer);
#endif

INSTANTIATE_CLASS(ROIAlignLayer);
REGISTER_LAYER_CLASS(ROIAlign);

}  // namespace caffe
