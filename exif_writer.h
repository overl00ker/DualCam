#ifndef EXIF_WRITER_H
#define EXIF_WRITER_H

#include <string>
#include <vector>
#include <cstdint>

struct ExifParams {
    std::string make;
    std::string model;
    std::string software;
    std::string description;
    uint32_t exposureTimeNum = 0;
    uint32_t exposureTimeDen = 0;
    uint16_t isoSpeed = 0;
};

std::vector<uint8_t> insertExif(const std::vector<uint8_t>& jpegData, const ExifParams& params);

std::string readExifDescription(const std::vector<uint8_t>& jpegData);

#endif
