

#include <stdio.h>
#include "ap_fixed.h"

void case_11(
    ap_int<16> in_data[7],
    ap_int<16> out_data[11]
)
{

#pragma HLS array_partition variable=in_data complete
#pragma HLS array_partition variable=out_data complete

    

ap_int<15> in1;
in1.range(14, 0) = in_data[0].range(14, 0);
ap_int<10> in2;
in2.range(9, 0) = in_data[1].range(9, 0);
ap_int<2> in3;
in3.range(1, 0) = in_data[2].range(1, 0);
ap_int<14> in4;
in4.range(13, 0) = in_data[3].range(13, 0);
ap_int<7> in5;
in5.range(6, 0) = in_data[4].range(6, 0);
ap_int<15> in6;
in6.range(14, 0) = in_data[5].range(14, 0);
ap_int<6> in7;
in7.range(5, 0) = in_data[6].range(5, 0);

ap_int<11> m8;
ap_int<16> m9;
ap_int<13> m10;
ap_int<11> m11;
ap_int<9> m12;
ap_int<13> m13;
ap_int<14> m14;
ap_int<13> m15;
ap_int<12> m16;
ap_int<13> m17;
ap_int<13> m18;
ap_int<8> m19;
ap_int<6> m20;
ap_int<11> m21;
ap_int<16> m22;
ap_int<13> m23;
ap_int<13> m24;
ap_int<14> m25;
ap_int<11> m26;
ap_int<16> m27;
ap_int<9> m28;
ap_int<16> m29;
ap_int<8> m30;
ap_int<13> m31;
ap_int<16> m32;
ap_int<7> m33;
ap_int<9> m34;
ap_int<14> m35;
ap_int<3> m36;
ap_int<16> m37;
ap_int<12> m38;
ap_int<7> m39;
ap_int<13> m40;
ap_int<16> m41;
ap_int<5> m42;
ap_int<15> m43;
ap_int<16> m44;
ap_int<9> m45;
ap_int<8> m46;
ap_int<9> m47;
ap_int<9> m48;
ap_int<7> m49;
ap_int<5> m50;
ap_int<13> m51;
ap_int<14> m52;
ap_int<9> m53;
ap_int<5> m54;
ap_int<8> m55;
ap_int<4> m56;
ap_int<7> m57;
ap_int<7> m58;
ap_int<5> m59;
ap_int<5> m60;
ap_int<9> m61;
ap_int<12> m62;
ap_int<13> m63;
ap_int<7> m64;
ap_int<8> m65;
ap_int<7> m66;
ap_int<4> m67;
ap_int<14> m68;
ap_int<7> m69;
ap_int<12> m70;
ap_int<13> m71;
ap_int<6> m72;

m8 = in1 * in2;
m9 = in3 * in4;
m10 = m8 * in7;
m11 = m9 + m8;
m12 = m8 * m9;
m13 = m9 * in7;
m14 = m10 * m11;
m15 = m10 * m10;
m16 = m12 * m13;
m17 = m11 * m15;
m18 = m15 * m16;
m19 = m12 * m13;
m20 = m16 + m16;
m21 = m20 * m17;
m22 = m17 * m15;
m23 = m19 * m16;
m24 = m22 + m22;
m25 = m21 * m20;
m26 = m24 + m24;
m27 = m23 + m26;
m28 = m26 * m27;
m29 = m22 + m24;
m30 = m26 * m29;
m31 = m30 * m25;
m32 = m27 * m31;
m33 = m26 * m32;
m34 = m30 * m33;
m35 = m32 * m33;
m36 = m33 * m33;
m37 = m34 + m33;
m38 = m35 * m37;
m39 = m33 * m34;
m40 = m34 + m39;
m41 = m40 * m38;
m42 = m36 * m36;
m43 = m41 * m39;
m44 = m43 * m42;
m45 = m40 * m41;
m46 = m40 * m39;
m47 = m40 * m44;
m48 = m42 * m42;
m49 = m46 + m43;
m50 = m46 * m43;
m51 = m50 + m44;
m52 = m47 * m46;
m53 = m51 * m47;
m54 = m50 * m53;
m55 = m51 * m52;
m56 = m54 * m49;
m57 = m54 * m52;
m58 = m52 * m51;
m59 = m55 * m56;
m60 = m59 * m56;
m61 = m58 + m54;
m62 = m55 + m60;
m63 = m61 + m58;
m64 = m57 + m60;
m65 = m59 + m59;
m66 = m59 + m65;
m67 = m60 + m64;
m68 = m63 * m64;
m69 = m65 * m67;
m70 = m65 * m63;
m71 = m68 * m64;
m72 = m67 * m67;

out_data[0] = m14;
out_data[1] = m18;
out_data[2] = m28;
out_data[3] = m45;
out_data[4] = m48;
out_data[5] = m62;
out_data[6] = m66;
out_data[7] = m69;
out_data[8] = m70;
out_data[9] = m71;
out_data[10] = m72;


}
    