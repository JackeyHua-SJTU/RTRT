#include "denoiser.h"
#include <iostream>

Denoiser::Denoiser() : m_useTemportal(false) {}

void Denoiser::Reprojection(const FrameInfo &frameInfo) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    Matrix4x4 preWorldToScreen =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 1];
    Matrix4x4 preWorldToCamera =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 2];
    
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Reproject
            if (frameInfo.m_id(x, y) >= 0.f) {
                auto id = frameInfo.m_id(x, y);
                auto inv_mat_cur = Inverse(frameInfo.m_matrix[id]);
                auto prev_M = m_preFrameInfo.m_matrix[id];
                auto operation_matrix = preWorldToScreen * prev_M * inv_mat_cur;
                auto p = operation_matrix(frameInfo.m_position(x, y), Float3::Point);
                if (p.x >= 0 && p.x < width && p.y >= 0 && p.y < height && id == m_preFrameInfo.m_id(p.x, p.y)) {
                    m_valid(x, y) = true;
                    m_misc(x, y) = m_accColor(p.x, p.y);
                }
            } else {
                m_valid(x, y) = false;
                m_misc(x, y) = Float3(0.f);
            }
        }
    }
    std::swap(m_misc, m_accColor);
}

void Denoiser::TemporalAccumulation(const Buffer2D<Float3> &curFilteredColor) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    int kernelRadius = 3;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (!m_valid(x, y)) {
                m_misc(x, y) = curFilteredColor(x, y);
                continue;
            }
            // TODO: Temporal clamp
            Float3 Ex(0.f), Ex2(0.f);
            for (int gapy = -kernelRadius; gapy <= kernelRadius; ++gapy) {
                for (int gapx = -kernelRadius; gapx <= kernelRadius; ++gapx) {
                    Ex += curFilteredColor(x + gapx, y + gapy);
                    Ex2 += Sqr(curFilteredColor(x + gapx, y + gapy));
                }
            }
            Float3 average = (Ex / Sqr(2 * kernelRadius + 1));
            Float3 dx = (Ex2 / Sqr(2 * kernelRadius + 1)) - Sqr(average);
            Float3 color = m_accColor(x, y);
            color = Clamp(color, average - SafeSqrt(dx) * m_colorBoxK, average + SafeSqrt(dx) * m_colorBoxK);
            // TODO: Exponential moving average
            m_misc(x, y) = Lerp(color, curFilteredColor(x, y), m_alpha);
        }
    }
    std::swap(m_misc, m_accColor);
}

Buffer2D<Float3> Denoiser::Filter(const FrameInfo &frameInfo) {
    int height = frameInfo.m_beauty.m_height;
    int width = frameInfo.m_beauty.m_width;
    Buffer2D<Float3> filteredImage = CreateBuffer2D<Float3>(width, height);
    int kernelRadius = 16;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Joint bilateral filter
            // TODO: Bonus ===> A Trous Wavelet
            // filteredImage(x, y) = frameInfo.m_beauty(x, y);
            double sum_weight = 0.f;
            Float3 sum_weight_val(0.f);
            for (int hb = y - kernelRadius; hb <= y + kernelRadius; ++hb) {
                for (int vb = x - kernelRadius; vb <= x + kernelRadius; ++vb) {
                    if (hb == y && vb == x) {
                        sum_weight += 1;
                        sum_weight_val += frameInfo.m_beauty(x, y);
                    }
                    double sigma_p = m_sigmaCoord, sigma_c = m_sigmaColor, sigma_n = m_sigmaNormal, sigma_d = m_sigmaPlane;
                    double d_pos = Sqr(y - hb) + Sqr(x - vb);
                    // std::cout << "dpos is " << d_pos << std::endl;
                    double d_color = SqrLength(frameInfo.m_beauty(vb, hb) - frameInfo.m_beauty(x, y));
                    // std::cout << "dcolor is " << d_color << std::endl;
                    double d_normal = Sqr(SafeAcos(Dot(frameInfo.m_normal(vb, hb), frameInfo.m_normal(x, y))));
                    // std::cout << "dnormal is " << d_normal << std::endl;
                    double d_plane = Sqr(Dot(frameInfo.m_normal(x, y), (frameInfo.m_position(vb, hb) - frameInfo.m_position(x, y)) /
                                         std::max(Length(frameInfo.m_position(vb, hb) - frameInfo.m_position(x, y)), 0.001f)));
                    // std::cout << "dplane is " << d_plane << std::endl;
                    double exponential = - (d_pos / (2.0f * Sqr(sigma_p)) + d_color / (2.0f * Sqr(sigma_c)) + d_normal / (2.0f * Sqr(sigma_n)) + 
                                                d_plane / (2.0f * Sqr(sigma_d)));
                    double w = std::exp(exponential);
                    // std::cout << "weight is " << sum_weight << std::endl;
                    sum_weight += w;
                    sum_weight_val += frameInfo.m_beauty(vb, hb) * w;
                }
            }
            // std::cout << "sum_weight is " << sum_weight << std::endl;
            // std::cout << "sum_weight_val is " << sum_weight_val << std::endl;
            filteredImage(x, y) = sum_weight_val / sum_weight;
        }
    }
    return filteredImage;
}

void Denoiser::Init(const FrameInfo &frameInfo, const Buffer2D<Float3> &filteredColor) {
    m_accColor.Copy(filteredColor);
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    m_misc = CreateBuffer2D<Float3>(width, height);
    m_valid = CreateBuffer2D<bool>(width, height);
}

void Denoiser::Maintain(const FrameInfo &frameInfo) { m_preFrameInfo = frameInfo; }

Buffer2D<Float3> Denoiser::ProcessFrame(const FrameInfo &frameInfo) {
    // Filter current frame
    Buffer2D<Float3> filteredColor;
    filteredColor = Filter(frameInfo);

    // std::cout << "In process\n";

    // Reproject previous frame color to current
    if (m_useTemportal) {
        // std::cout << "Before reprojection\n";
        Reprojection(frameInfo);
        // std::cout << "After reprojection\n";
        TemporalAccumulation(filteredColor);
    } else {
        // std::cout << "Before init\n";
        Init(frameInfo, filteredColor);
    }

    // Maintain
    Maintain(frameInfo);
    if (!m_useTemportal) {
        m_useTemportal = true;
    }
    return m_accColor;
}
