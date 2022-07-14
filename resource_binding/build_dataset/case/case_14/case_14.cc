

#include <stdio.h>
#include "ap_fixed.h"

void case_14(
    ap_int<16> in_data[6],
    ap_int<16> out_data[6]
)
{

#pragma HLS array_partition variable=in_data complete
#pragma HLS array_partition variable=out_data complete

    

ap_int<5> in1;
in1.range(4, 0) = in_data[0].range(4, 0);
ap_int<13> in2;
in2.range(12, 0) = in_data[1].range(12, 0);
ap_int<14> in3;
in3.range(13, 0) = in_data[2].range(13, 0);
ap_int<6> in4;
in4.range(5, 0) = in_data[3].range(5, 0);
ap_int<8> in5;
in5.range(7, 0) = in_data[4].range(7, 0);
ap_int<15> in6;
in6.range(14, 0) = in_data[5].range(14, 0);

ap_int<16> m7;
ap_int<14> m8;
ap_int<10> m9;
ap_int<14> m10;
ap_int<13> m11;
ap_int<15> m12;
ap_int<12> m13;
ap_int<6> m14;
ap_int<14> m15;
ap_int<6> m16;
ap_int<10> m17;
ap_int<10> m18;
ap_int<6> m19;
ap_int<6> m20;
ap_int<8> m21;
ap_int<7> m22;
ap_int<12> m23;
ap_int<6> m24;
ap_int<7> m25;
ap_int<8> m26;
ap_int<10> m27;
ap_int<16> m28;
ap_int<14> m29;
ap_int<14> m30;
ap_int<10> m31;
ap_int<13> m32;
ap_int<16> m33;
ap_int<9> m34;
ap_int<6> m35;
ap_int<16> m36;
ap_int<9> m37;
ap_int<11> m38;
ap_int<4> m39;
ap_int<7> m40;
ap_int<15> m41;
ap_int<14> m42;
ap_int<7> m43;
ap_int<10> m44;
ap_int<12> m45;
ap_int<8> m46;
ap_int<16> m47;
ap_int<11> m48;
ap_int<16> m49;
ap_int<11> m50;

m7 = in1 * in6;
m8 = m7 * in6;
m9 = in6 * m7;
m10 = in6 + in4;
m11 = in6 * in6;
m12 = in6 * m10;
m13 = m7 + m9;
m14 = m11 * m9;
m15 = m10 * m12;
m16 = m14 * m14;
m17 = m14 * m11;
m18 = m16 * m14;
m19 = m15 * m13;
m20 = m18 + m17;
m21 = m15 * m15;
m22 = m20 * m17;
m23 = m17 * m20;
m24 = m22 * m21;
m25 = m19 * m21;
m26 = m20 * m21;
m27 = m22 * m25;
m28 = m27 * m25;
m29 = m23 + m26;
m30 = m28 * m28;
m31 = m29 * m27;
m32 = m26 * m30;
m33 = m28 + m32;
m34 = m29 * m33;
m35 = m29 * m31;
m36 = m34 * m30;
m37 = m31 * m33;
m38 = m32 + m32;
m39 = m38 * m35;
m40 = m36 + m35;
m41 = m36 * m36;
m42 = m41 * m38;
m43 = m37 * m39;
m44 = m41 * m42;
m45 = m41 * m41;
m46 = m42 * m45;
m47 = m44 * m45;
m48 = m42 * m43;
m49 = m46 * m45;
m50 = m49 * m44;

out_data[0] = m8;
out_data[1] = m24;
out_data[2] = m40;
out_data[3] = m47;
out_data[4] = m48;
out_data[5] = m50;


}
    