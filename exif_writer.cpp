#include "exif_writer.h"
#include <cstring>
#include <cstdint>

namespace {
    void write16(std::vector<uint8_t>& buf, uint16_t val) {
        buf.push_back(val & 0xFF);
        buf.push_back((val >> 8) & 0xFF);
    }

    void write32(std::vector<uint8_t>& buf, uint32_t val) {
        buf.push_back(val & 0xFF);
        buf.push_back((val >> 8) & 0xFF);
        buf.push_back((val >> 16) & 0xFF);
        buf.push_back((val >> 24) & 0xFF);
    }

    struct IfdEntry {
        uint16_t tag;
        uint16_t format;
        uint32_t count;
        uint32_t dataOrOffset;
    };

    void writeEntry(std::vector<uint8_t>& buf, const IfdEntry& e) {
        write16(buf, e.tag);
        write16(buf, e.format);
        write32(buf, e.count);
        write32(buf, e.dataOrOffset);
    }
}

std::vector<uint8_t> insertExif(const std::vector<uint8_t>& jpegData, const ExifParams& params) {
    if (jpegData.size() < 2 || jpegData[0] != 0xFF || jpegData[1] != 0xD8) {
        return jpegData;
    }

    std::vector<uint8_t> tiffData;

    tiffData.push_back('I'); tiffData.push_back('I');
    write16(tiffData, 42);
    write32(tiffData, 8);

    std::vector<IfdEntry> ifd0;
    std::vector<IfdEntry> subIfd;
    std::vector<uint8_t> extraData;

    auto addString = [&](std::vector<IfdEntry>& ifd, uint16_t tag, const std::string& str) {
        if (str.empty()) return;
        IfdEntry e;
        e.tag = tag;
        e.format = 2;
        e.count = str.length() + 1;
        if (e.count <= 4) {
            e.dataOrOffset = 0;
            memcpy(&e.dataOrOffset, str.c_str(), e.count);
        } else {
            e.dataOrOffset = extraData.size();
            extraData.insert(extraData.end(), str.begin(), str.end());
            extraData.push_back(0);
        }
        ifd.push_back(e);
    };

    addString(ifd0, 0x010E, params.description);
    addString(ifd0, 0x010F, params.make);
    addString(ifd0, 0x0110, params.model);
    addString(ifd0, 0x0131, params.software);

    if (params.exposureTimeDen > 0) {
        IfdEntry e;
        e.tag = 0x829A;
        e.format = 5;
        e.count = 1;
        e.dataOrOffset = extraData.size();
        write32(extraData, params.exposureTimeNum);
        write32(extraData, params.exposureTimeDen);
        subIfd.push_back(e);
    }
    if (params.isoSpeed > 0) {
        IfdEntry e;
        e.tag = 0x8827;
        e.format = 3;
        e.count = 1;
        e.dataOrOffset = params.isoSpeed;
        subIfd.push_back(e);
    }

    uint32_t exifOffsetPlaceholderIdx = 0;
    if (!subIfd.empty()) {
        IfdEntry e;
        e.tag = 0x8769;
        e.format = 4;
        e.count = 1;
        e.dataOrOffset = 0;
        exifOffsetPlaceholderIdx = ifd0.size();
        ifd0.push_back(e);
    }

    uint32_t ifd0Offset = 8;
    uint32_t ifd0Size = 2 + ifd0.size() * 12 + 4;

    uint32_t subIfdOffset = ifd0Offset + ifd0Size;
    uint32_t subIfdSize = subIfd.empty() ? 0 : (2 + subIfd.size() * 12 + 4);

    uint32_t extraDataOffset = subIfdOffset + subIfdSize;

    for (auto& e : ifd0) {
        if (e.format == 2 && e.count > 4) {
            e.dataOrOffset += extraDataOffset;
        }
    }
    for (auto& e : subIfd) {
        if (e.format == 5) {
            e.dataOrOffset += extraDataOffset;
        }
    }
    if (!subIfd.empty()) {
        ifd0[exifOffsetPlaceholderIdx].dataOrOffset = subIfdOffset;
    }

    write16(tiffData, ifd0.size());
    for (const auto& e : ifd0) writeEntry(tiffData, e);
    write32(tiffData, 0);

    if (!subIfd.empty()) {
        write16(tiffData, subIfd.size());
        for (const auto& e : subIfd) writeEntry(tiffData, e);
        write32(tiffData, 0);
    }

    tiffData.insert(tiffData.end(), extraData.begin(), extraData.end());

    std::vector<uint8_t> res;
    res.push_back(0xFF);
    res.push_back(0xD8);

    res.push_back(0xFF);
    res.push_back(0xE1);
    uint16_t app1Len = 2 + 6 + tiffData.size();
    res.push_back((app1Len >> 8) & 0xFF);
    res.push_back(app1Len & 0xFF);

    res.push_back('E'); res.push_back('x'); res.push_back('i'); res.push_back('f');
    res.push_back(0); res.push_back(0);

    res.insert(res.end(), tiffData.begin(), tiffData.end());

    res.insert(res.end(), jpegData.begin() + 2, jpegData.end());

    return res;
}

namespace {
    uint16_t read16le(const uint8_t* p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }
    uint32_t read32le(const uint8_t* p) { return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24); }
    uint16_t read16be(const uint8_t* p) { return uint16_t(p[1]) | (uint16_t(p[0]) << 8); }
    uint32_t read32be(const uint8_t* p) { return uint32_t(p[3]) | (uint32_t(p[2]) << 8) | (uint32_t(p[1]) << 16) | (uint32_t(p[0]) << 24); }
}

std::string readExifDescription(const std::vector<uint8_t>& jpegData) {
    if (jpegData.size() < 4 || jpegData[0] != 0xFF || jpegData[1] != 0xD8) return {};

    size_t i = 2;
    while (i + 4 < jpegData.size()) {
        if (jpegData[i] != 0xFF) return {};
        uint8_t marker = jpegData[i + 1];
        if (marker == 0xD9 || marker == 0xDA) return {};
        uint16_t segLen = (uint16_t(jpegData[i + 2]) << 8) | jpegData[i + 3];
        if (segLen < 2 || i + 2 + segLen > jpegData.size()) return {};

        if (marker == 0xE1 && segLen >= 8 &&
            jpegData[i + 4] == 'E' && jpegData[i + 5] == 'x' &&
            jpegData[i + 6] == 'i' && jpegData[i + 7] == 'f' &&
            jpegData[i + 8] == 0  && jpegData[i + 9] == 0) {
            const uint8_t* tiff = jpegData.data() + i + 10;
            size_t tiffLen = segLen - 2 - 6;
            if (tiffLen < 8) return {};

            bool little;
            if (tiff[0] == 'I' && tiff[1] == 'I') little = true;
            else if (tiff[0] == 'M' && tiff[1] == 'M') little = false;
            else return {};

            auto r16 = [&](const uint8_t* p) { return little ? read16le(p) : read16be(p); };
            auto r32 = [&](const uint8_t* p) { return little ? read32le(p) : read32be(p); };

            if (r16(tiff + 2) != 42) return {};
            uint32_t ifd0Off = r32(tiff + 4);
            if (ifd0Off + 2 > tiffLen) return {};
            uint16_t nEntries = r16(tiff + ifd0Off);
            if (ifd0Off + 2 + nEntries * 12u > tiffLen) return {};

            for (uint16_t k = 0; k < nEntries; ++k) {
                const uint8_t* e = tiff + ifd0Off + 2 + k * 12;
                uint16_t tag = r16(e);
                uint16_t fmt = r16(e + 2);
                uint32_t cnt = r32(e + 4);
                if (tag == 0x010E && fmt == 2 && cnt > 0) {
                    if (cnt <= 4) {
                        std::string s(reinterpret_cast<const char*>(e + 8), cnt);
                        if (!s.empty() && s.back() == '\0') s.pop_back();
                        return s;
                    }
                    uint32_t off = r32(e + 8);
                    if (off + cnt > tiffLen) return {};
                    std::string s(reinterpret_cast<const char*>(tiff + off), cnt);
                    if (!s.empty() && s.back() == '\0') s.pop_back();
                    return s;
                }
            }
            return {};
        }

        i += 2 + segLen;
    }
    return {};
}
