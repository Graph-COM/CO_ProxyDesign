

#include <stdio.h>
#include "ap_fixed.h"

void case_1(
    ap_int<16> in_data[7],
    ap_int<16> out_data[4]
)
{

#pragma HLS array_partition variable=in_data complete
#pragma HLS array_partition variable=out_data complete

    

ap_int<14> in1;
in1.range(13, 0) = in_data[0].range(13, 0);
ap_int<9> in2;
in2.range(8, 0) = in_data[1].range(8, 0);
ap_int<5> in3;
in3.range(4, 0) = in_data[2].range(4, 0);
ap_int<14> in4;
in4.range(13, 0) = in_data[3].range(13, 0);
ap_int<15> in5;
in5.range(14, 0) = in_data[4].range(14, 0);
ap_int<10> in6;
in6.range(9, 0) = in_data[5].range(9, 0);
ap_int<5> in7;
in7.range(4, 0) = in_data[6].range(4, 0);

ap_int<13> m8;
ap_int<5> m9;
ap_int<15> m10;
ap_int<12> m11;
ap_int<9> m12;
ap_int<6> m13;
ap_int<7> m14;
ap_int<5> m15;
ap_int<8> m16;
ap_int<9> m17;
ap_int<10> m18;
ap_int<12> m19;
ap_int<9> m20;
ap_int<10> m21;
ap_int<10> m22;
ap_int<15> m23;
ap_int<16> m24;
ap_int<16> m25;
ap_int<5> m26;
ap_int<12> m27;
ap_int<13> m28;
ap_int<9> m29;
ap_int<10> m30;
ap_int<12> m31;
ap_int<15> m32;
ap_int<13> m33;
ap_int<12> m34;
ap_int<6> m35;
ap_int<10> m36;
ap_int<10> m37;
ap_int<15> m38;
ap_int<6> m39;
ap_int<9> m40;
ap_int<10> m41;
ap_int<14> m42;
ap_int<4> m43;

m8 = in2 * in1;
m9 = in3 * in5;
m10 = in3 + in6;
m11 = in6 * in5;
m12 = m9 * m8;
m13 = in6 + m10;
m14 = m8 * m8;
m15 = m9 * m11;
m16 = m13 + m12;
m17 = m13 * m15;
m18 = m17 + m13;
m19 = m15 * m14;
m20 = m18 * m14;
m21 = m16 * m18;
m22 = m15 * m19;
m23 = m20 * m21;
m24 = m21 + m17;
m25 = m18 * m19;
m26 = m22 * m19;
m27 = m25 * m20;
m28 = m21 * m24;
m29 = m25 * m25;
m30 = m27 + m23;
m31 = m24 * m29;
m32 = m26 * m31;
m33 = m29 * m31;
m34 = m33 * m31;
m35 = m34 * m28;
m36 = m30 * m32;
m37 = m30 * m30;
m38 = m36 + m35;
m39 = m36 * m35;
m40 = m34 * m39;
m41 = m37 * m39;
m42 = m36 * m39;
m43 = m40 * m36;

out_data[0] = m38;
out_data[1] = m41;
out_data[2] = m42;
out_data[3] = m43;


}
    