

#include <stdio.h>
#include "ap_fixed.h"

void case_7(
    ap_int<16> in_data[5],
    ap_int<16> out_data[9]
)
{

#pragma HLS array_partition variable=in_data complete
#pragma HLS array_partition variable=out_data complete

    

ap_int<7> in1;
in1.range(6, 0) = in_data[0].range(6, 0);
ap_int<15> in2;
in2.range(14, 0) = in_data[1].range(14, 0);
ap_int<15> in3;
in3.range(14, 0) = in_data[2].range(14, 0);
ap_int<14> in4;
in4.range(13, 0) = in_data[3].range(13, 0);
ap_int<12> in5;
in5.range(11, 0) = in_data[4].range(11, 0);

ap_int<16> m6;
ap_int<14> m7;
ap_int<8> m8;
ap_int<11> m9;
ap_int<15> m10;
ap_int<12> m11;
ap_int<6> m12;
ap_int<11> m13;
ap_int<14> m14;
ap_int<12> m15;
ap_int<6> m16;
ap_int<12> m17;
ap_int<9> m18;
ap_int<5> m19;
ap_int<3> m20;
ap_int<9> m21;
ap_int<8> m22;
ap_int<12> m23;
ap_int<12> m24;
ap_int<4> m25;
ap_int<12> m26;
ap_int<8> m27;
ap_int<6> m28;
ap_int<15> m29;
ap_int<9> m30;
ap_int<4> m31;
ap_int<15> m32;
ap_int<16> m33;
ap_int<16> m34;
ap_int<7> m35;
ap_int<13> m36;
ap_int<14> m37;
ap_int<11> m38;
ap_int<13> m39;
ap_int<14> m40;
ap_int<10> m41;
ap_int<16> m42;
ap_int<6> m43;
ap_int<11> m44;
ap_int<10> m45;
ap_int<13> m46;
ap_int<8> m47;
ap_int<16> m48;
ap_int<6> m49;

m6 = in2 * in2;
m7 = in3 * in4;
m8 = m6 * in5;
m9 = m8 * in4;
m10 = m9 * m6;
m11 = m10 * m9;
m12 = m8 * m7;
m13 = m12 * m12;
m14 = m9 * m10;
m15 = m10 * m14;
m16 = m12 * m12;
m17 = m12 * m12;
m18 = m16 * m17;
m19 = m18 * m17;
m20 = m19 * m19;
m21 = m18 * m20;
m22 = m18 * m19;
m23 = m20 + m21;
m24 = m23 * m22;
m25 = m20 * m20;
m26 = m24 * m24;
m27 = m23 * m22;
m28 = m27 * m23;
m29 = m24 * m25;
m30 = m28 * m29;
m31 = m27 + m30;
m32 = m30 * m28;
m33 = m29 * m29;
m34 = m32 * m31;
m35 = m30 + m30;
m36 = m35 * m34;
m37 = m35 + m32;
m38 = m33 * m33;
m39 = m37 * m38;
m40 = m37 * m38;
m41 = m37 * m37;
m42 = m38 + m40;
m43 = m39 * m38;
m44 = m41 * m40;
m45 = m44 * m43;
m46 = m43 + m45;
m47 = m44 * m43;
m48 = m47 + m44;
m49 = m44 * m44;

out_data[0] = m11;
out_data[1] = m13;
out_data[2] = m15;
out_data[3] = m26;
out_data[4] = m36;
out_data[5] = m42;
out_data[6] = m46;
out_data[7] = m48;
out_data[8] = m49;


}
    