

#include <stdio.h>
#include "ap_fixed.h"

void case_3(
    ap_int<16> in_data[9],
    ap_int<16> out_data[9]
)
{

#pragma HLS array_partition variable=in_data complete
#pragma HLS array_partition variable=out_data complete

    

ap_int<2> in1;
in1.range(1, 0) = in_data[0].range(1, 0);
ap_int<8> in2;
in2.range(7, 0) = in_data[1].range(7, 0);
ap_int<7> in3;
in3.range(6, 0) = in_data[2].range(6, 0);
ap_int<10> in4;
in4.range(9, 0) = in_data[3].range(9, 0);
ap_int<14> in5;
in5.range(13, 0) = in_data[4].range(13, 0);
ap_int<13> in6;
in6.range(12, 0) = in_data[5].range(12, 0);
ap_int<9> in7;
in7.range(8, 0) = in_data[6].range(8, 0);
ap_int<4> in8;
in8.range(3, 0) = in_data[7].range(3, 0);
ap_int<15> in9;
in9.range(14, 0) = in_data[8].range(14, 0);

ap_int<10> m10;
ap_int<8> m11;
ap_int<15> m12;
ap_int<6> m13;
ap_int<9> m14;
ap_int<11> m15;
ap_int<6> m16;
ap_int<6> m17;
ap_int<13> m18;
ap_int<15> m19;
ap_int<16> m20;
ap_int<16> m21;
ap_int<16> m22;
ap_int<9> m23;
ap_int<14> m24;
ap_int<8> m25;
ap_int<5> m26;
ap_int<10> m27;
ap_int<8> m28;
ap_int<15> m29;
ap_int<10> m30;
ap_int<9> m31;
ap_int<10> m32;
ap_int<12> m33;
ap_int<13> m34;
ap_int<11> m35;
ap_int<9> m36;
ap_int<8> m37;
ap_int<7> m38;
ap_int<13> m39;
ap_int<6> m40;
ap_int<13> m41;
ap_int<10> m42;
ap_int<10> m43;
ap_int<15> m44;
ap_int<12> m45;
ap_int<12> m46;
ap_int<6> m47;
ap_int<15> m48;
ap_int<11> m49;
ap_int<8> m50;
ap_int<7> m51;
ap_int<9> m52;
ap_int<10> m53;
ap_int<7> m54;
ap_int<10> m55;
ap_int<6> m56;
ap_int<16> m57;
ap_int<13> m58;
ap_int<4> m59;
ap_int<16> m60;
ap_int<8> m61;
ap_int<9> m62;

m10 = in5 + in4;
m11 = in3 * in7;
m12 = in8 * in5;
m13 = in7 + m12;
m14 = in9 * in7;
m15 = in7 * m13;
m16 = in9 * in7;
m17 = m16 * m15;
m18 = m10 + m12;
m19 = m14 * m17;
m20 = m14 * m18;
m21 = m14 * m19;
m22 = m16 * m18;
m23 = m16 * m19;
m24 = m15 * m16;
m25 = m18 + m16;
m26 = m23 + m18;
m27 = m21 * m21;
m28 = m20 * m24;
m29 = m22 * m23;
m30 = m28 * m25;
m31 = m26 * m29;
m32 = m24 + m28;
m33 = m29 * m29;
m34 = m33 * m26;
m35 = m32 * m29;
m36 = m30 * m31;
m37 = m34 + m29;
m38 = m34 * m31;
m39 = m32 * m34;
m40 = m33 + m32;
m41 = m36 + m38;
m42 = m34 * m40;
m43 = m37 * m42;
m44 = m38 * m39;
m45 = m37 * m39;
m46 = m42 + m42;
m47 = m38 + m40;
m48 = m46 * m40;
m49 = m41 * m41;
m50 = m44 * m48;
m51 = m43 * m44;
m52 = m48 * m50;
m53 = m47 * m47;
m54 = m53 * m47;
m55 = m51 * m46;
m56 = m55 * m53;
m57 = m51 * m49;
m58 = m53 * m49;
m59 = m52 * m55;
m60 = m57 * m58;
m61 = m57 * m54;
m62 = m53 * m54;

out_data[0] = m11;
out_data[1] = m27;
out_data[2] = m35;
out_data[3] = m45;
out_data[4] = m56;
out_data[5] = m59;
out_data[6] = m60;
out_data[7] = m61;
out_data[8] = m62;


}
    