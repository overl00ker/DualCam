#include "exif_writer.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdio>
#include <string>
#include <vector>

int main()
{
    const cv::Mat img(48, 64, CV_8UC3, cv::Scalar(30, 120, 200));
    std::vector<uint8_t> jpeg;
    if (!cv::imencode(".jpg", img, jpeg)) {
        std::puts("FAIL: imencode");
        return 1;
    }

    ExifParams p;
    p.make = "DualCamQt";
    p.model = "dual_combined";
    p.software = "DualCam Analysis Tool";
    p.description = R"({"mode":"dual_combined","noiseFloor":15,"userName":"probe"})";
    p.exposureTimeNum = 10000;
    p.exposureTimeDen = 1000000;
    p.isoSpeed = 200;

    bool ok = true;
    const std::vector<uint8_t> out = insertExif(jpeg, p);
    const std::string back = readExifDescription(out);
    if (back != p.description) {
        std::printf("FAIL: description read back as '%s'\n", back.c_str());
        ok = false;
    }
    if (cv::imdecode(out, cv::IMREAD_COLOR).size() != img.size()) {
        std::puts("FAIL: JPEG with EXIF does not decode");
        ok = false;
    }
    if (!readExifDescription(jpeg).empty()) {
        std::puts("FAIL: EXIF found in a plain JPEG");
        ok = false;
    }
    p.description = "abc";
    if (readExifDescription(insertExif(jpeg, p)) != "abc") {
        std::puts("FAIL: inline (<= 4 byte) description");
        ok = false;
    }

    std::puts(ok ? "exif round trip: OK" : "exif round trip: FAILED");
    return ok ? 0 : 1;
}
