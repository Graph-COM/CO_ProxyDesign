

#include <stdio.h>
#include "ap_fixed.h"

void case_8(
    ap_int<16> in_data[11],
    ap_int<16> out_data[14]
)
{

#pragma HLS array_partition variable=in_data complete
#pragma HLS array_partition variable=out_data complete

    

ap_int<11> in1;
in1.range(10, 0) = in_data[0].range(10, 0);
ap_int<2> in2;
in2.range(1, 0) = in_data[1].range(1, 0);
ap_int<5> in3;
in3.range(4, 0) = in_data[2].range(4, 0);
ap_int<3> in4;
in4.range(2, 0) = in_data[3].range(2, 0);
ap_int<9> in5;
in5.range(8, 0) = in_data[4].range(8, 0);
ap_int<16> in6;
in6.range(15, 0) = in_data[5].range(15, 0);
ap_int<5> in7;
in7.range(4, 0) = in_data[6].range(4, 0);
ap_int<5> in8;
in8.range(4, 0) = in_data[7].range(4, 0);
ap_int<9> in9;
in9.range(8, 0) = in_data[8].range(8, 0);
ap_int<5> in10;
in10.range(4, 0) = in_data[9].range(4, 0);
ap_int<14> in11;
in11.range(13, 0) = in_data[10].range(13, 0);

ap_int<5> m12;
ap_int<4> m13;
ap_int<11> m14;
ap_int<15> m15;
ap_int<16> m16;
ap_int<7> m17;
ap_int<9> m18;
ap_int<15> m19;
ap_int<7> m20;
ap_int<9> m21;
ap_int<9> m22;
ap_int<15> m23;
ap_int<8> m24;
ap_int<14> m25;
ap_int<15> m26;
ap_int<5> m27;
ap_int<13> m28;
ap_int<16> m29;
ap_int<7> m30;
ap_int<4> m31;
ap_int<16> m32;
ap_int<15> m33;
ap_int<16> m34;
ap_int<7> m35;
ap_int<8> m36;
ap_int<10> m37;
ap_int<9> m38;
ap_int<13> m39;
ap_int<13> m40;
ap_int<14> m41;
ap_int<8> m42;
ap_int<6> m43;
ap_int<6> m44;
ap_int<10> m45;
ap_int<8> m46;
ap_int<10> m47;
ap_int<9> m48;
ap_int<6> m49;
ap_int<6> m50;
ap_int<6> m51;
ap_int<6> m52;
ap_int<12> m53;
ap_int<8> m54;
ap_int<9> m55;
ap_int<6> m56;
ap_int<10> m57;
ap_int<7> m58;
ap_int<6> m59;
ap_int<6> m60;
ap_int<13> m61;
ap_int<6> m62;
ap_int<6> m63;
ap_int<9> m64;
ap_int<9> m65;
ap_int<3> m66;
ap_int<10> m67;
ap_int<9> m68;
ap_int<15> m69;
ap_int<7> m70;
ap_int<13> m71;
ap_int<16> m72;
ap_int<3> m73;
ap_int<12> m74;
ap_int<7> m75;
ap_int<6> m76;
ap_int<5> m77;
ap_int<6> m78;
ap_int<10> m79;
ap_int<10> m80;
ap_int<15> m81;
ap_int<9> m82;
ap_int<6> m83;
ap_int<10> m84;
ap_int<11> m85;
ap_int<7> m86;
ap_int<15> m87;
ap_int<10> m88;
ap_int<13> m89;
ap_int<16> m90;
ap_int<4> m91;
ap_int<6> m92;
ap_int<10> m93;
ap_int<6> m94;
ap_int<8> m95;
ap_int<7> m96;
ap_int<9> m97;

m12 = in2 + in4;
m13 = in9 + in8;
m14 = in11 * in11;
m15 = in5 * in5;
m16 = in6 * m12;
m17 = m13 + m14;
m18 = m14 * in11;
m19 = m16 * in11;
m20 = in10 + m12;
m21 = m13 * m18;
m22 = m12 * m20;
m23 = m21 * m21;
m24 = m17 * m19;
m25 = m24 * m21;
m26 = m18 * m20;
m27 = m19 + m20;
m28 = m22 * m21;
m29 = m22 * m25;
m30 = m25 * m19;
m31 = m25 + m27;
m32 = m30 * m28;
m33 = m25 * m31;
m34 = m28 * m23;
m35 = m26 * m34;
m36 = m34 + m30;
m37 = m31 * m33;
m38 = m29 * m35;
m39 = m30 * m33;
m40 = m39 * m33;
m41 = m36 * m36;
m42 = m31 * m41;
m43 = m39 * m36;
m44 = m36 + m41;
m45 = m35 * m43;
m46 = m39 * m42;
m47 = m37 * m39;
m48 = m42 + m46;
m49 = m43 * m40;
m50 = m40 * m44;
m51 = m45 * m48;
m52 = m41 + m42;
m53 = m49 * m45;
m54 = m46 + m45;
m55 = m45 + m48;
m56 = m51 * m46;
m57 = m49 * m53;
m58 = m47 * m47;
m59 = m55 + m55;
m60 = m59 + m56;
m61 = m50 * m58;
m62 = m58 * m60;
m63 = m57 * m54;
m64 = m58 * m55;
m65 = m61 * m56;
m66 = m62 + m60;
m67 = m56 * m58;
m68 = m60 * m61;
m69 = m61 * m66;
m70 = m59 * m67;
m71 = m67 * m69;
m72 = m69 * m67;
m73 = m70 * m70;
m74 = m73 * m65;
m75 = m74 * m70;
m76 = m73 + m65;
m77 = m76 * m73;
m78 = m69 * m67;
m79 = m76 * m76;
m80 = m70 * m76;
m81 = m74 * m79;
m82 = m74 * m75;
m83 = m76 * m73;
m84 = m73 * m74;
m85 = m84 * m74;
m86 = m85 + m77;
m87 = m79 * m80;
m88 = m77 * m85;
m89 = m81 * m78;
m90 = m82 * m87;
m91 = m83 * m88;
m92 = m85 * m85;
m93 = m85 + m89;
m94 = m83 + m90;
m95 = m92 + m84;
m96 = m89 * m86;
m97 = m94 * m88;

out_data[0] = m15;
out_data[1] = m32;
out_data[2] = m38;
out_data[3] = m52;
out_data[4] = m63;
out_data[5] = m64;
out_data[6] = m68;
out_data[7] = m71;
out_data[8] = m72;
out_data[9] = m91;
out_data[10] = m93;
out_data[11] = m95;
out_data[12] = m96;
out_data[13] = m97;


}
    