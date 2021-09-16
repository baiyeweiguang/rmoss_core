// Copyright 2020 RoboMaster-OSS
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef RM_CAM__CAM_INTERFACE_HPP_
#define RM_CAM__CAM_INTERFACE_HPP_

#include "opencv2/opencv.hpp"

namespace rm_cam
{

enum class CamParamType
{
  Width,
  Height,
  AutoExposure,
  Exposure,
  Brightness,
  AutoWhiteBalance,
  WhiteBalance,
  Gain,
  Gamma,
  Contrast,
  Saturation,
  Hue,
  Fps
};

// common interface for camera device (usb cam,virtual cam,etc.)
class CamInterface
{
public:
  virtual bool open() = 0;
  virtual void close() = 0;
  virtual bool is_open() = 0;
  virtual bool grab_image(cv::Mat & imgae) = 0;
  // set and get parameter interface (optional)
  virtual bool set_parameter(CamParamType /*type*/, int /*value*/)
  {
    return false;
  }
  virtual bool get_parameter(CamParamType /*type*/, int & /*value*/)
  {
    return false;
  }
};
}  // namespace rm_cam

#endif  // RM_CAM__CAM_INTERFACE_HPP_