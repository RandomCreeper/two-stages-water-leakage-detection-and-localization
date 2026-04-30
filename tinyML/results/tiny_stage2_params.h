#pragma once
#include <stdint.h>
#define TINY_STAGE2_BINARY_FEATURES 30
static const int16_t TINY_STAGE2_BINARY_COEF_Q15[30] = {
    6, 6, -47, -144, -144, 383, -145, 24, 6, -664, -38, -17,
    -243, -17, -177, 61, -152, -69, 81, 94, -18, 2, 783, 127,
    1, -2, 2684, 1461, -11687, 32767
};

static const float TINY_STAGE2_BINARY_COEF_SCALE = 0.00036051277f;

static const int16_t TINY_STAGE2_BINARY_BIAS_Q15[1] = {
    -32767
};

static const float TINY_STAGE2_BINARY_BIAS_SCALE = 0.00023801512f;

static const float TINY_STAGE2_BINARY_THRESHOLD_PROB = 0.91775782f;

static const float TINY_STAGE2_BINARY_THRESHOLD_LOGIT = 2.4122652f;

#define TINY_STAGE2_NUM_PIPES 14
#define TINY_STAGE2_LOC_FEATURES 40
static const int16_t TINY_STAGE2_PIPE_1_BIN_COEF_Q15[40] = {
    79, -183, 782, 121, 343, -127, -366, 155, -327, 357, 458, -34,
    -610, -7, -2087, 638, 127, -812, -618, -408, -72, 2292, -1057, 892,
    236, 75, -31, -23535, -135, -202, 114, -9719, 165, 112, 517, 32767,
    -98, -796, -796, 677
};

static const float TINY_STAGE2_PIPE_1_BIN_COEF_SCALE = 0.00020318008f;

static const int16_t TINY_STAGE2_PIPE_1_BIN_BIAS_Q15[1] = {
    -32767
};

static const float TINY_STAGE2_PIPE_1_BIN_BIAS_SCALE = 8.6852287e-05f;

static const int16_t TINY_STAGE2_PIPE_2_BIN_COEF_Q15[40] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0
};

static const float TINY_STAGE2_PIPE_2_BIN_COEF_SCALE = 1f;

static const int16_t TINY_STAGE2_PIPE_2_BIN_BIAS_Q15[1] = {
    0
};

static const float TINY_STAGE2_PIPE_2_BIN_BIAS_SCALE = 1f;

static const int16_t TINY_STAGE2_PIPE_3_BIN_COEF_Q15[40] = {
    63, 2012, 213, 1060, -510, -1442, 453, 621, -694, 1501, -1679, 1404,
    -2531, 47, -1220, -1901, -566, 996, -2800, 1336, -3228, -1019, -1578, -1954,
    22, -336, -234, -32767, 177, -239, -284, -24228, 482, 55, 1283, 23346,
    76, 1561, 1561, -142
};

static const float TINY_STAGE2_PIPE_3_BIN_COEF_SCALE = 7.701663e-05f;

static const int16_t TINY_STAGE2_PIPE_3_BIN_BIAS_Q15[1] = {
    -32767
};

static const float TINY_STAGE2_PIPE_3_BIN_BIAS_SCALE = 5.4622592e-05f;

static const int16_t TINY_STAGE2_PIPE_4_BIN_COEF_Q15[40] = {
    -4872, -1282, 4151, 182, 2630, 1616, 1508, -1898, -5494, 9190, 393, 712,
    -230, -4449, -5602, 3405, 3875, 16499, 4096, 836, -6581, -12819, -11473, -4108,
    -1507, 995, 29, 32767, -492, 1106, 179, 47, -922, 726, -352, 104,
    -100, 1565, 1565, -641
};

static const float TINY_STAGE2_PIPE_4_BIN_COEF_SCALE = 6.2134074e-05f;

static const int16_t TINY_STAGE2_PIPE_4_BIN_BIAS_Q15[1] = {
    32767
};

static const float TINY_STAGE2_PIPE_4_BIN_BIAS_SCALE = 4.0347848e-05f;

static const int16_t TINY_STAGE2_PIPE_5_BIN_COEF_Q15[40] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0
};

static const float TINY_STAGE2_PIPE_5_BIN_COEF_SCALE = 1f;

static const int16_t TINY_STAGE2_PIPE_5_BIN_BIAS_Q15[1] = {
    0
};

static const float TINY_STAGE2_PIPE_5_BIN_BIAS_SCALE = 1f;

static const int16_t TINY_STAGE2_PIPE_6_BIN_COEF_Q15[40] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0
};

static const float TINY_STAGE2_PIPE_6_BIN_COEF_SCALE = 1f;

static const int16_t TINY_STAGE2_PIPE_6_BIN_BIAS_Q15[1] = {
    0
};

static const float TINY_STAGE2_PIPE_6_BIN_BIAS_SCALE = 1f;

static const int16_t TINY_STAGE2_PIPE_7_BIN_COEF_Q15[40] = {
    2874, -1077, -2037, -40, -1945, -373, 892, 612, 1195, -1071, -773, -269,
    78, -770, 1551, 170, -1844, -3515, -3433, -331, 1597, 5504, 4924, 1243,
    531, 40, -246, -32767, -101, -431, -135, -5850, 345, -50, 854, 32578,
    -98, 72, 72, 2178
};

static const float TINY_STAGE2_PIPE_7_BIN_COEF_SCALE = 0.00011867926f;

static const int16_t TINY_STAGE2_PIPE_7_BIN_BIAS_Q15[1] = {
    -32767
};

static const float TINY_STAGE2_PIPE_7_BIN_BIAS_SCALE = 0.00011725097f;

static const int16_t TINY_STAGE2_PIPE_8_BIN_COEF_Q15[40] = {
    -3444, 1322, 174, -603, 764, -203, 6160, -2808, -3692, 3319, -5128, -452,
    -1829, 2996, 7879, 1308, -1190, 7065, -4483, -202, -5278, -2778, 4390, -5185,
    -1512, 790, 114, 32767, -548, 1207, -188, 5367, -1359, 879, -228, -30866,
    17, 2055, 2055, -1476
};

static const float TINY_STAGE2_PIPE_8_BIN_COEF_SCALE = 6.15649e-05f;

static const int16_t TINY_STAGE2_PIPE_8_BIN_BIAS_Q15[1] = {
    32767
};

static const float TINY_STAGE2_PIPE_8_BIN_BIAS_SCALE = 4.1392446e-05f;

static const int16_t TINY_STAGE2_PIPE_9_BIN_COEF_Q15[40] = {
    4784, -14954, 9395, 3464, 16147, 8893, -16603, -6757, -2079, -16457, 11310, 2979,
    17816, 8192, -18745, -7339, -1991, -32767, 26596, 6715, 24526, 3612, -15180, 3968,
    -457, 2491, -950, -7429, -1892, 988, -992, 12717, -2498, 1234, -2330, -21047,
    248, 9075, 9075, 2652
};

static const float TINY_STAGE2_PIPE_9_BIN_COEF_SCALE = 8.5484128e-06f;

static const int16_t TINY_STAGE2_PIPE_9_BIN_BIAS_Q15[1] = {
    -32767
};

static const float TINY_STAGE2_PIPE_9_BIN_BIAS_SCALE = 0.00015456844f;

static const int16_t TINY_STAGE2_PIPE_10_BIN_COEF_Q15[40] = {
    1636, 340, 5125, -1396, 2250, -4954, -6651, -1855, 5377, 7343, 6654, 1878,
    2440, -17284, -7602, 402, 1105, 16603, 5653, 226, 1368, -70, -6659, -8338,
    176, 2096, -215, -11910, -1893, 517, -265, 32767, -2207, 1584, -806, -7912,
    410, -552, -552, 778
};

static const float TINY_STAGE2_PIPE_10_BIN_COEF_SCALE = 3.243429e-05f;

static const int16_t TINY_STAGE2_PIPE_10_BIN_BIAS_Q15[1] = {
    -32767
};

static const float TINY_STAGE2_PIPE_10_BIN_BIAS_SCALE = 9.7243973e-05f;

static const int16_t TINY_STAGE2_PIPE_11_BIN_COEF_Q15[40] = {
    113, -419, 581, 105, 241, 376, -240, 205, -221, 52, 422, -5,
    -453, 70, -1339, 74, 207, -122, -440, -215, -165, 1169, -941, 600,
    204, 63, -59, -23214, -107, -173, 176, -10753, 146, 112, 481, 32767,
    -88, -566, -566, 386
};

static const float TINY_STAGE2_PIPE_11_BIN_COEF_SCALE = 0.00023724045f;

static const int16_t TINY_STAGE2_PIPE_11_BIN_BIAS_Q15[1] = {
    -32767
};

static const float TINY_STAGE2_PIPE_11_BIN_BIAS_SCALE = 9.6408886e-05f;

static const int16_t TINY_STAGE2_PIPE_12_BIN_COEF_Q15[40] = {
    -10494, -4130, -7036, -5836, 3453, 13070, 10445, 547, -2901, -3320, 1757, -5621,
    1147, 4040, 23004, -3356, -13646, -7142, -6773, -9846, 427, 17678, 32767, -1534,
    -250, 1668, -854, 26270, -1815, 233, -2333, 27098, -589, 1345, 276, 13523,
    -680, -13346, -13346, 10295
};

static const float TINY_STAGE2_PIPE_12_BIN_COEF_SCALE = 1.9190806e-05f;

static const int16_t TINY_STAGE2_PIPE_12_BIN_BIAS_Q15[1] = {
    -32767
};

static const float TINY_STAGE2_PIPE_12_BIN_BIAS_SCALE = 0.00012346058f;

static const int16_t TINY_STAGE2_PIPE_13_BIN_COEF_Q15[40] = {
    -76, 152, 681, 393, 790, -87, -147, -50, -573, -331, -364, 104,
    -547, -469, -312, 490, -640, -347, -741, 19, -649, 243, -280, 448,
    16, -20, -5, -27291, -5, -78, -73, -16304, 172, 74, 925, 32767,
    -11, -251, -251, -186
};

static const float TINY_STAGE2_PIPE_13_BIN_COEF_SCALE = 0.00024124224f;

static const int16_t TINY_STAGE2_PIPE_13_BIN_BIAS_Q15[1] = {
    -32767
};

static const float TINY_STAGE2_PIPE_13_BIN_BIAS_SCALE = 8.712294e-05f;

static const int16_t TINY_STAGE2_PIPE_14_BIN_COEF_Q15[40] = {
    1357, 749, -38, -187, -2582, 2038, -911, -1237, 1142, -1010, 1528, 950,
    589, -397, 685, -3058, 184, -3425, 799, 2541, 3319, -479, 1698, -97,
    2701, -2230, 29, 23521, 1541, -2238, -158, 32767, 2287, -2531, -485, -20485,
    32, -2556, -2556, 3873
};

static const float TINY_STAGE2_PIPE_14_BIN_COEF_SCALE = 4.425769e-05f;

static const int16_t TINY_STAGE2_PIPE_14_BIN_BIAS_Q15[1] = {
    -32767
};

static const float TINY_STAGE2_PIPE_14_BIN_BIAS_SCALE = 0.00012559218f;
