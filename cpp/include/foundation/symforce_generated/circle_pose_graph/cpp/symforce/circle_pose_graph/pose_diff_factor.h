// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <sym/pose3.h>

#include <Eigen/Core>

namespace circle_pose_graph {

/**
 * This function was autogenerated from a symbolic function. Do not modify by
 * hand.
 *
 * Symbolic function: pose_diff_factor
 *
 * Args:
 *     pose1: Pose3
 *     pose2: Pose3
 *     estimated_movement: Pose3
 *     epsilon: Scalar
 *
 * Outputs:
 *     res: Matrix61
 *     jacobian: (6x12) jacobian of res wrt args pose1 (6), pose2 (6)
 *     hessian: (12x12) Gauss-Newton hessian for args pose1 (6), pose2 (6)
 *     rhs: (12x1) Gauss-Newton rhs for args pose1 (6), pose2 (6)
 */
template <typename Scalar>
void PoseDiffFactor(const sym::Pose3<Scalar>& pose1,
                    const sym::Pose3<Scalar>& pose2,
                    const sym::Pose3<Scalar>& estimated_movement,
                    const Scalar epsilon,
                    Eigen::Matrix<Scalar, 6, 1>* const res = nullptr,
                    Eigen::Matrix<Scalar, 6, 12>* const jacobian = nullptr,
                    Eigen::Matrix<Scalar, 12, 12>* const hessian = nullptr,
                    Eigen::Matrix<Scalar, 12, 1>* const rhs = nullptr) {
  // Total ops: 1234

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _pose1 = pose1.Data();
  const Eigen::Matrix<Scalar, 7, 1>& _pose2 = pose2.Data();
  const Eigen::Matrix<Scalar, 7, 1>& _estimated_movement =
      estimated_movement.Data();

  // Intermediate terms (299)
  const Scalar _tmp0 = _estimated_movement[0] * _pose1[2];
  const Scalar _tmp1 = _estimated_movement[2] * _pose1[0];
  const Scalar _tmp2 = _estimated_movement[1] * _pose1[3];
  const Scalar _tmp3 = _estimated_movement[3] * _pose1[1];
  const Scalar _tmp4 = -_tmp0 + _tmp1 - _tmp2 - _tmp3;
  const Scalar _tmp5 = _pose2[2] * _tmp4;
  const Scalar _tmp6 = _estimated_movement[1] * _pose1[2];
  const Scalar _tmp7 = _estimated_movement[3] * _pose1[0];
  const Scalar _tmp8 = _estimated_movement[0] * _pose1[3];
  const Scalar _tmp9 = _estimated_movement[2] * _pose1[1];
  const Scalar _tmp10 = _tmp6 - _tmp7 - _tmp8 - _tmp9;
  const Scalar _tmp11 = _pose2[3] * _tmp10;
  const Scalar _tmp12 = _estimated_movement[3] * _pose1[2];
  const Scalar _tmp13 = _estimated_movement[1] * _pose1[0];
  const Scalar _tmp14 = _estimated_movement[2] * _pose1[3];
  const Scalar _tmp15 = _estimated_movement[0] * _pose1[1];
  const Scalar _tmp16 = -_tmp12 - _tmp13 - _tmp14 + _tmp15;
  const Scalar _tmp17 = _pose2[1] * _tmp16;
  const Scalar _tmp18 = _estimated_movement[3] * _pose1[3];
  const Scalar _tmp19 = _estimated_movement[2] * _pose1[2];
  const Scalar _tmp20 = _estimated_movement[0] * _pose1[0];
  const Scalar _tmp21 = _estimated_movement[1] * _pose1[1];
  const Scalar _tmp22 = _tmp18 - _tmp19 - _tmp20 - _tmp21;
  const Scalar _tmp23 = _pose2[0] * _tmp22;
  const Scalar _tmp24 = _tmp11 - _tmp17 + _tmp23 + _tmp5;
  const Scalar _tmp25 = _pose2[1] * _tmp4;
  const Scalar _tmp26 = _pose2[0] * _tmp10;
  const Scalar _tmp27 = _pose2[2] * _tmp16;
  const Scalar _tmp28 = _tmp25 + _tmp26 + _tmp27;
  const Scalar _tmp29 = _pose2[3] * _tmp22;
  const Scalar _tmp30 = -_tmp29;
  const Scalar _tmp31 = 1 - epsilon;
  const Scalar _tmp32 = std::min<Scalar>(_tmp31, std::fabs(_tmp28 + _tmp30));
  const Scalar _tmp33 = 2 * std::min<Scalar>(0, (((-_tmp28 + _tmp29) > 0) -
                                                 ((-_tmp28 + _tmp29) < 0))) +
                        1;
  const Scalar _tmp34 = 2 * _tmp33;
  const Scalar _tmp35 = _tmp34 * std::acos(_tmp32) /
                        std::sqrt(Scalar(1 - std::pow(_tmp32, Scalar(2))));
  const Scalar _tmp36 = _tmp24 * _tmp35;
  const Scalar _tmp37 = _pose2[3] * _tmp4;
  const Scalar _tmp38 = _pose2[2] * _tmp10;
  const Scalar _tmp39 = _pose2[0] * _tmp16;
  const Scalar _tmp40 = _pose2[1] * _tmp22;
  const Scalar _tmp41 = _tmp37 - _tmp38 + _tmp39 + _tmp40;
  const Scalar _tmp42 = _tmp35 * _tmp41;
  const Scalar _tmp43 = _pose2[0] * _tmp4;
  const Scalar _tmp44 = _pose2[1] * _tmp10;
  const Scalar _tmp45 = _pose2[3] * _tmp16;
  const Scalar _tmp46 = _pose2[2] * _tmp22;
  const Scalar _tmp47 = -_tmp43 + _tmp44 + _tmp45 + _tmp46;
  const Scalar _tmp48 = _tmp35 * _tmp47;
  const Scalar _tmp49 = 2 * _tmp4;
  const Scalar _tmp50 = _tmp10 * _tmp49;
  const Scalar _tmp51 = 2 * _tmp16;
  const Scalar _tmp52 = _tmp22 * _tmp51;
  const Scalar _tmp53 = _tmp50 - _tmp52;
  const Scalar _tmp54 = _tmp10 * _tmp51;
  const Scalar _tmp55 = _tmp22 * _tmp49;
  const Scalar _tmp56 = _tmp54 + _tmp55;
  const Scalar _tmp57 = -2 * std::pow(_tmp4, Scalar(2));
  const Scalar _tmp58 = 1 - 2 * std::pow(_tmp16, Scalar(2));
  const Scalar _tmp59 = _tmp57 + _tmp58;
  const Scalar _tmp60 = 2 * _estimated_movement[3];
  const Scalar _tmp61 = _estimated_movement[2] * _tmp60;
  const Scalar _tmp62 = 2 * _estimated_movement[1];
  const Scalar _tmp63 = _estimated_movement[0] * _tmp62;
  const Scalar _tmp64 = _tmp61 + _tmp63;
  const Scalar _tmp65 = _estimated_movement[3] * _tmp62;
  const Scalar _tmp66 = 2 * _estimated_movement[0] * _estimated_movement[2];
  const Scalar _tmp67 = -_tmp65 + _tmp66;
  const Scalar _tmp68 = -2 * std::pow(_estimated_movement[1], Scalar(2));
  const Scalar _tmp69 = 1 - 2 * std::pow(_estimated_movement[2], Scalar(2));
  const Scalar _tmp70 = _tmp68 + _tmp69;
  const Scalar _tmp71 = std::pow(_pose1[2], Scalar(2));
  const Scalar _tmp72 = -2 * _tmp71;
  const Scalar _tmp73 = std::pow(_pose1[1], Scalar(2));
  const Scalar _tmp74 = -2 * _tmp73;
  const Scalar _tmp75 = _tmp72 + _tmp74 + 1;
  const Scalar _tmp76 = 2 * _pose1[2];
  const Scalar _tmp77 = _pose1[0] * _tmp76;
  const Scalar _tmp78 = 2 * _pose1[1] * _pose1[3];
  const Scalar _tmp79 = -_tmp78;
  const Scalar _tmp80 = _tmp77 + _tmp79;
  const Scalar _tmp81 = 2 * _pose1[0];
  const Scalar _tmp82 = _pose1[1] * _tmp81;
  const Scalar _tmp83 = _pose1[3] * _tmp76;
  const Scalar _tmp84 = _tmp82 + _tmp83;
  const Scalar _tmp85 = _pose1[5] * _tmp84 + _pose1[6] * _tmp80;
  const Scalar _tmp86 = _pose1[4] * _tmp75 + _tmp85;
  const Scalar _tmp87 = std::pow(_pose1[0], Scalar(2));
  const Scalar _tmp88 = 1 - 2 * _tmp87;
  const Scalar _tmp89 = _tmp74 + _tmp88;
  const Scalar _tmp90 = _tmp77 + _tmp78;
  const Scalar _tmp91 = _pose1[3] * _tmp81;
  const Scalar _tmp92 = -_tmp91;
  const Scalar _tmp93 = _pose1[1] * _tmp76;
  const Scalar _tmp94 = _tmp92 + _tmp93;
  const Scalar _tmp95 = _pose1[4] * _tmp90 + _pose1[5] * _tmp94;
  const Scalar _tmp96 = _pose1[6] * _tmp89 + _tmp95;
  const Scalar _tmp97 = _tmp72 + _tmp88;
  const Scalar _tmp98 = -_tmp83;
  const Scalar _tmp99 = _tmp82 + _tmp98;
  const Scalar _tmp100 = _tmp91 + _tmp93;
  const Scalar _tmp101 = _pose1[4] * _tmp99 + _pose1[6] * _tmp100;
  const Scalar _tmp102 = _pose1[5] * _tmp97 + _tmp101;
  const Scalar _tmp103 = -_estimated_movement[4] * _tmp70 -
                         _estimated_movement[5] * _tmp64 -
                         _estimated_movement[6] * _tmp67 + _pose2[4] * _tmp59 +
                         _pose2[5] * _tmp53 + _pose2[6] * _tmp56 -
                         _tmp102 * _tmp64 - _tmp67 * _tmp96 - _tmp70 * _tmp86;
  const Scalar _tmp104 = _tmp50 + _tmp52;
  const Scalar _tmp105 = _tmp16 * _tmp49;
  const Scalar _tmp106 = 2 * _tmp10;
  const Scalar _tmp107 = _tmp106 * _tmp22;
  const Scalar _tmp108 = _tmp105 - _tmp107;
  const Scalar _tmp109 = -2 * std::pow(_tmp10, Scalar(2));
  const Scalar _tmp110 = _tmp109 + _tmp58;
  const Scalar _tmp111 = -_tmp61 + _tmp63;
  const Scalar _tmp112 = -2 * std::pow(_estimated_movement[0], Scalar(2));
  const Scalar _tmp113 = _tmp112 + _tmp69;
  const Scalar _tmp114 = _estimated_movement[0] * _tmp60;
  const Scalar _tmp115 = _estimated_movement[2] * _tmp62;
  const Scalar _tmp116 = _tmp114 + _tmp115;
  const Scalar _tmp117 =
      -_estimated_movement[4] * _tmp111 - _estimated_movement[5] * _tmp113 -
      _estimated_movement[6] * _tmp116 + _pose2[4] * _tmp104 +
      _pose2[5] * _tmp110 + _pose2[6] * _tmp108 - _tmp102 * _tmp113 -
      _tmp111 * _tmp86 - _tmp116 * _tmp96;
  const Scalar _tmp118 = _tmp105 + _tmp107;
  const Scalar _tmp119 = _tmp54 - _tmp55;
  const Scalar _tmp120 = _tmp109 + _tmp57 + 1;
  const Scalar _tmp121 = _tmp65 + _tmp66;
  const Scalar _tmp122 = _tmp112 + _tmp68 + 1;
  const Scalar _tmp123 = -_tmp114 + _tmp115;
  const Scalar _tmp124 =
      -_estimated_movement[4] * _tmp121 - _estimated_movement[5] * _tmp123 -
      _estimated_movement[6] * _tmp122 + _pose2[4] * _tmp119 +
      _pose2[5] * _tmp118 + _pose2[6] * _tmp120 - _tmp102 * _tmp123 -
      _tmp121 * _tmp86 - _tmp122 * _tmp96;
  const Scalar _tmp125 = (Scalar(1) / Scalar(2)) * _tmp6;
  const Scalar _tmp126 = (Scalar(1) / Scalar(2)) * _tmp7;
  const Scalar _tmp127 = -_tmp126;
  const Scalar _tmp128 = (Scalar(1) / Scalar(2)) * _tmp8;
  const Scalar _tmp129 = -_tmp128;
  const Scalar _tmp130 = (Scalar(1) / Scalar(2)) * _tmp9;
  const Scalar _tmp131 = -_tmp125 + _tmp127 + _tmp129 + _tmp130;
  const Scalar _tmp132 = (Scalar(1) / Scalar(2)) * _tmp1;
  const Scalar _tmp133 = (Scalar(1) / Scalar(2)) * _tmp3;
  const Scalar _tmp134 = (Scalar(1) / Scalar(2)) * _tmp0;
  const Scalar _tmp135 = (Scalar(1) / Scalar(2)) * _tmp2;
  const Scalar _tmp136 = _tmp134 - _tmp135;
  const Scalar _tmp137 = _tmp132 + _tmp133 + _tmp136;
  const Scalar _tmp138 = (Scalar(1) / Scalar(2)) * _tmp12;
  const Scalar _tmp139 = -_tmp138;
  const Scalar _tmp140 = (Scalar(1) / Scalar(2)) * _tmp13;
  const Scalar _tmp141 = (Scalar(1) / Scalar(2)) * _tmp14;
  const Scalar _tmp142 = (Scalar(1) / Scalar(2)) * _tmp15;
  const Scalar _tmp143 = _tmp139 + _tmp140 + _tmp141 + _tmp142;
  const Scalar _tmp144 = (Scalar(1) / Scalar(2)) * _tmp19;
  const Scalar _tmp145 = -_tmp144;
  const Scalar _tmp146 = (Scalar(1) / Scalar(2)) * _tmp20;
  const Scalar _tmp147 = -Scalar(1) / Scalar(2) * _tmp18;
  const Scalar _tmp148 = (Scalar(1) / Scalar(2)) * _tmp21;
  const Scalar _tmp149 = _tmp147 - _tmp148;
  const Scalar _tmp150 = _tmp145 + _tmp146 + _tmp149;
  const Scalar _tmp151 = _tmp28 + _tmp30;
  const Scalar _tmp152 = std::fabs(_tmp151);
  const Scalar _tmp153 = std::min<Scalar>(_tmp152, _tmp31);
  const Scalar _tmp154 = std::acos(_tmp153);
  const Scalar _tmp155 = 1 - std::pow(_tmp153, Scalar(2));
  const Scalar _tmp156 = _tmp34 / std::sqrt(_tmp155);
  const Scalar _tmp157 = _tmp154 * _tmp156;
  const Scalar _tmp158 = _pose2[0] * _tmp150 + _pose2[1] * _tmp143 +
                         _pose2[2] * _tmp137 - _pose2[3] * _tmp131;
  const Scalar _tmp159 =
      _tmp33 * ((((-_tmp152 + _tmp31) > 0) - ((-_tmp152 + _tmp31) < 0)) + 1) *
      (((_tmp151) > 0) - ((_tmp151) < 0));
  const Scalar _tmp160 = _tmp153 * _tmp159 / (_tmp155 * std::sqrt(_tmp155));
  const Scalar _tmp161 = _tmp154 * _tmp160;
  const Scalar _tmp162 = _tmp161 * _tmp24;
  const Scalar _tmp163 = _tmp159 / _tmp155;
  const Scalar _tmp164 = _tmp163 * _tmp24;
  const Scalar _tmp165 = _tmp157 * (_pose2[0] * _tmp131 - _pose2[1] * _tmp137 +
                                    _pose2[2] * _tmp143 + _pose2[3] * _tmp150) +
                         _tmp158 * _tmp162 - _tmp158 * _tmp164;
  const Scalar _tmp166 = _tmp161 * _tmp41;
  const Scalar _tmp167 = _tmp163 * _tmp41;
  const Scalar _tmp168 = _tmp157 * (_pose2[0] * _tmp137 + _pose2[1] * _tmp131 -
                                    _pose2[2] * _tmp150 + _pose2[3] * _tmp143) +
                         _tmp158 * _tmp166 - _tmp158 * _tmp167;
  const Scalar _tmp169 = _tmp163 * _tmp47;
  const Scalar _tmp170 = _tmp161 * _tmp47;
  const Scalar _tmp171 = _tmp157 * (-_pose2[0] * _tmp143 + _pose2[1] * _tmp150 +
                                    _pose2[2] * _tmp131 + _pose2[3] * _tmp137) -
                         _tmp158 * _tmp169 + _tmp158 * _tmp170;
  const Scalar _tmp172 = 4 * _tmp16;
  const Scalar _tmp173 = -_tmp137 * _tmp172;
  const Scalar _tmp174 = 4 * _tmp4;
  const Scalar _tmp175 = -_tmp143 * _tmp174;
  const Scalar _tmp176 = _tmp131 * _tmp51;
  const Scalar _tmp177 = 2 * _tmp22;
  const Scalar _tmp178 = _tmp137 * _tmp177;
  const Scalar _tmp179 = 2 * _tmp150;
  const Scalar _tmp180 = _tmp106 * _tmp143 + _tmp179 * _tmp4;
  const Scalar _tmp181 = _tmp131 * _tmp49;
  const Scalar _tmp182 = _tmp143 * _tmp177;
  const Scalar _tmp183 = _tmp106 * _tmp137 + _tmp16 * _tmp179;
  const Scalar _tmp184 = -_tmp73;
  const Scalar _tmp185 = _tmp184 + _tmp71;
  const Scalar _tmp186 = -_tmp87;
  const Scalar _tmp187 = std::pow(_pose1[3], Scalar(2));
  const Scalar _tmp188 = _tmp186 + _tmp187;
  const Scalar _tmp189 = _pose1[6] * (_tmp185 + _tmp188) + _tmp95;
  const Scalar _tmp190 = -_tmp82;
  const Scalar _tmp191 = -_tmp187;
  const Scalar _tmp192 = _tmp191 + _tmp87;
  const Scalar _tmp193 = -_tmp93;
  const Scalar _tmp194 = _pose1[4] * (_tmp190 + _tmp83) +
                         _pose1[5] * (_tmp185 + _tmp192) +
                         _pose1[6] * (_tmp193 + _tmp92);
  const Scalar _tmp195 = _pose2[4] * (_tmp173 + _tmp175) +
                         _pose2[5] * (-_tmp176 - _tmp178 + _tmp180) +
                         _pose2[6] * (_tmp181 + _tmp182 + _tmp183) -
                         _tmp189 * _tmp64 - _tmp194 * _tmp67;
  const Scalar _tmp196 = _tmp106 * _tmp131;
  const Scalar _tmp197 = _tmp179 * _tmp22;
  const Scalar _tmp198 = _tmp137 * _tmp49 + _tmp143 * _tmp51;
  const Scalar _tmp199 = 4 * _tmp10;
  const Scalar _tmp200 = -_tmp150 * _tmp199;
  const Scalar _tmp201 = _pose2[4] * (_tmp176 + _tmp178 + _tmp180) +
                         _pose2[5] * (_tmp173 + _tmp200) +
                         _pose2[6] * (-_tmp196 - _tmp197 + _tmp198) -
                         _tmp113 * _tmp189 - _tmp116 * _tmp194;
  const Scalar _tmp202 = _pose2[4] * (-_tmp181 - _tmp182 + _tmp183) +
                         _pose2[5] * (_tmp196 + _tmp197 + _tmp198) +
                         _pose2[6] * (_tmp175 + _tmp200) - _tmp122 * _tmp194 -
                         _tmp123 * _tmp189;
  const Scalar _tmp203 = -_tmp133;
  const Scalar _tmp204 = -_tmp132 + _tmp136 + _tmp203;
  const Scalar _tmp205 = _tmp125 + _tmp130;
  const Scalar _tmp206 = _tmp127 + _tmp128 + _tmp205;
  const Scalar _tmp207 = _tmp140 - _tmp141;
  const Scalar _tmp208 = _tmp138 + _tmp142 + _tmp207;
  const Scalar _tmp209 = -_tmp146;
  const Scalar _tmp210 = _tmp145 + _tmp147 + _tmp148 + _tmp209;
  const Scalar _tmp211 = _pose2[0] * _tmp208 + _pose2[1] * _tmp210 +
                         _pose2[2] * _tmp206 - _pose2[3] * _tmp204;
  const Scalar _tmp212 = _tmp157 * (_pose2[0] * _tmp204 - _pose2[1] * _tmp206 +
                                    _pose2[2] * _tmp210 + _pose2[3] * _tmp208) +
                         _tmp162 * _tmp211 - _tmp164 * _tmp211;
  const Scalar _tmp213 = _tmp157 * (_pose2[0] * _tmp206 + _pose2[1] * _tmp204 -
                                    _pose2[2] * _tmp208 + _pose2[3] * _tmp210) +
                         _tmp166 * _tmp211 - _tmp167 * _tmp211;
  const Scalar _tmp214 = _tmp157 * (-_pose2[0] * _tmp210 + _pose2[1] * _tmp208 +
                                    _pose2[2] * _tmp204 + _pose2[3] * _tmp206) -
                         _tmp169 * _tmp211 + _tmp170 * _tmp211;
  const Scalar _tmp215 = _tmp204 * _tmp49;
  const Scalar _tmp216 = _tmp177 * _tmp210;
  const Scalar _tmp217 = _tmp106 * _tmp206 + _tmp208 * _tmp51;
  const Scalar _tmp218 = 2 * _tmp204;
  const Scalar _tmp219 = _tmp16 * _tmp218;
  const Scalar _tmp220 = _tmp177 * _tmp206;
  const Scalar _tmp221 = _tmp106 * _tmp210 + _tmp208 * _tmp49;
  const Scalar _tmp222 = -_tmp174 * _tmp210;
  const Scalar _tmp223 = -_tmp172 * _tmp206;
  const Scalar _tmp224 = -_tmp71;
  const Scalar _tmp225 =
      _pose1[4] * (_tmp184 + _tmp187 + _tmp224 + _tmp87) + _tmp85;
  const Scalar _tmp226 = -_tmp77;
  const Scalar _tmp227 = _tmp224 + _tmp73;
  const Scalar _tmp228 = _pose1[4] * (_tmp226 + _tmp79) +
                         _pose1[5] * (_tmp193 + _tmp91) +
                         _pose1[6] * (_tmp192 + _tmp227);
  const Scalar _tmp229 = _pose2[4] * (_tmp222 + _tmp223) +
                         _pose2[5] * (-_tmp219 - _tmp220 + _tmp221) +
                         _pose2[6] * (_tmp215 + _tmp216 + _tmp217) -
                         _tmp225 * _tmp67 - _tmp228 * _tmp70;
  const Scalar _tmp230 = -_tmp199 * _tmp208;
  const Scalar _tmp231 = _tmp10 * _tmp218;
  const Scalar _tmp232 = _tmp177 * _tmp208;
  const Scalar _tmp233 = _tmp206 * _tmp49 + _tmp210 * _tmp51;
  const Scalar _tmp234 = _pose2[4] * (_tmp219 + _tmp220 + _tmp221) +
                         _pose2[5] * (_tmp223 + _tmp230) +
                         _pose2[6] * (-_tmp231 - _tmp232 + _tmp233) -
                         _tmp111 * _tmp228 - _tmp116 * _tmp225;
  const Scalar _tmp235 = _pose2[4] * (-_tmp215 - _tmp216 + _tmp217) +
                         _pose2[5] * (_tmp231 + _tmp232 + _tmp233) +
                         _pose2[6] * (_tmp222 + _tmp230) - _tmp121 * _tmp228 -
                         _tmp122 * _tmp225;
  const Scalar _tmp236 = _tmp132 + _tmp134 + _tmp135 + _tmp203;
  const Scalar _tmp237 = _tmp126 + _tmp129 + _tmp205;
  const Scalar _tmp238 = _tmp139 - _tmp142 + _tmp207;
  const Scalar _tmp239 = _tmp144 + _tmp149 + _tmp209;
  const Scalar _tmp240 = _pose2[0] * _tmp236 + _pose2[1] * _tmp237 +
                         _pose2[2] * _tmp239 - _pose2[3] * _tmp238;
  const Scalar _tmp241 = _tmp157 * (_pose2[0] * _tmp238 - _pose2[1] * _tmp239 +
                                    _pose2[2] * _tmp237 + _pose2[3] * _tmp236) +
                         _tmp162 * _tmp240 - _tmp164 * _tmp240;
  const Scalar _tmp242 = _tmp157 * (_pose2[0] * _tmp239 + _pose2[1] * _tmp238 -
                                    _pose2[2] * _tmp236 + _pose2[3] * _tmp237) +
                         _tmp166 * _tmp240 - _tmp167 * _tmp240;
  const Scalar _tmp243 = _tmp157 * (-_pose2[0] * _tmp237 + _pose2[1] * _tmp236 +
                                    _pose2[2] * _tmp238 + _pose2[3] * _tmp239) -
                         _tmp169 * _tmp240 + _tmp170 * _tmp240;
  const Scalar _tmp244 = _tmp238 * _tmp51;
  const Scalar _tmp245 = 2 * _tmp239;
  const Scalar _tmp246 = _tmp22 * _tmp245;
  const Scalar _tmp247 = _tmp106 * _tmp237 + _tmp236 * _tmp49;
  const Scalar _tmp248 = _tmp238 * _tmp49;
  const Scalar _tmp249 = _tmp177 * _tmp237;
  const Scalar _tmp250 = _tmp10 * _tmp245 + _tmp236 * _tmp51;
  const Scalar _tmp251 = -_tmp172 * _tmp239;
  const Scalar _tmp252 = -_tmp174 * _tmp237;
  const Scalar _tmp253 = _pose1[4] * (_tmp186 + _tmp191 + _tmp71 + _tmp73) +
                         _pose1[5] * (_tmp190 + _tmp98) +
                         _pose1[6] * (_tmp226 + _tmp78);
  const Scalar _tmp254 = _pose1[5] * (_tmp188 + _tmp227) + _tmp101;
  const Scalar _tmp255 = _pose2[4] * (_tmp251 + _tmp252) +
                         _pose2[5] * (-_tmp244 - _tmp246 + _tmp247) +
                         _pose2[6] * (_tmp248 + _tmp249 + _tmp250) -
                         _tmp253 * _tmp64 - _tmp254 * _tmp70;
  const Scalar _tmp256 = _tmp106 * _tmp238;
  const Scalar _tmp257 = _tmp177 * _tmp236;
  const Scalar _tmp258 = _tmp237 * _tmp51 + _tmp245 * _tmp4;
  const Scalar _tmp259 = -_tmp199 * _tmp236;
  const Scalar _tmp260 = _pose2[4] * (_tmp244 + _tmp246 + _tmp247) +
                         _pose2[5] * (_tmp251 + _tmp259) +
                         _pose2[6] * (-_tmp256 - _tmp257 + _tmp258) -
                         _tmp111 * _tmp254 - _tmp113 * _tmp253;
  const Scalar _tmp261 = _pose2[4] * (-_tmp248 - _tmp249 + _tmp250) +
                         _pose2[5] * (_tmp256 + _tmp257 + _tmp258) +
                         _pose2[6] * (_tmp252 + _tmp259) - _tmp121 * _tmp254 -
                         _tmp123 * _tmp253;
  const Scalar _tmp262 = -_tmp64 * _tmp99 - _tmp67 * _tmp90 - _tmp70 * _tmp75;
  const Scalar _tmp263 =
      -_tmp111 * _tmp75 - _tmp113 * _tmp99 - _tmp116 * _tmp90;
  const Scalar _tmp264 =
      -_tmp121 * _tmp75 - _tmp122 * _tmp90 - _tmp123 * _tmp99;
  const Scalar _tmp265 = -_tmp64 * _tmp97 - _tmp67 * _tmp94 - _tmp70 * _tmp84;
  const Scalar _tmp266 =
      -_tmp111 * _tmp84 - _tmp113 * _tmp97 - _tmp116 * _tmp94;
  const Scalar _tmp267 =
      -_tmp121 * _tmp84 - _tmp122 * _tmp94 - _tmp123 * _tmp97;
  const Scalar _tmp268 = -_tmp100 * _tmp64 - _tmp67 * _tmp89 - _tmp70 * _tmp80;
  const Scalar _tmp269 =
      -_tmp100 * _tmp113 - _tmp111 * _tmp80 - _tmp116 * _tmp89;
  const Scalar _tmp270 =
      -_tmp100 * _tmp123 - _tmp121 * _tmp80 - _tmp122 * _tmp89;
  const Scalar _tmp271 =
      _tmp157 *
      (-Scalar(1) / Scalar(2) * _tmp25 - Scalar(1) / Scalar(2) * _tmp26 -
       Scalar(1) / Scalar(2) * _tmp27 + (Scalar(1) / Scalar(2)) * _tmp29);
  const Scalar _tmp272 = (Scalar(1) / Scalar(2)) * _tmp5;
  const Scalar _tmp273 = (Scalar(1) / Scalar(2)) * _tmp11;
  const Scalar _tmp274 = (Scalar(1) / Scalar(2)) * _tmp17;
  const Scalar _tmp275 = (Scalar(1) / Scalar(2)) * _tmp23;
  const Scalar _tmp276 = _tmp272 + _tmp273 - _tmp274 + _tmp275;
  const Scalar _tmp277 = _tmp162 * _tmp276 - _tmp164 * _tmp276 + _tmp271;
  const Scalar _tmp278 = (Scalar(1) / Scalar(2)) * _tmp43;
  const Scalar _tmp279 = (Scalar(1) / Scalar(2)) * _tmp44;
  const Scalar _tmp280 = (Scalar(1) / Scalar(2)) * _tmp45;
  const Scalar _tmp281 = (Scalar(1) / Scalar(2)) * _tmp46;
  const Scalar _tmp282 = -_tmp278 + _tmp279 + _tmp280 + _tmp281;
  const Scalar _tmp283 =
      _tmp157 * _tmp282 + _tmp166 * _tmp276 - _tmp167 * _tmp276;
  const Scalar _tmp284 = (Scalar(1) / Scalar(2)) * _tmp37;
  const Scalar _tmp285 = (Scalar(1) / Scalar(2)) * _tmp38;
  const Scalar _tmp286 = (Scalar(1) / Scalar(2)) * _tmp39;
  const Scalar _tmp287 = (Scalar(1) / Scalar(2)) * _tmp40;
  const Scalar _tmp288 = _tmp157 * (-_tmp284 + _tmp285 - _tmp286 - _tmp287) -
                         _tmp169 * _tmp276 + _tmp170 * _tmp276;
  const Scalar _tmp289 = _tmp284 - _tmp285 + _tmp286 + _tmp287;
  const Scalar _tmp290 = _tmp154 * _tmp289;
  const Scalar _tmp291 = _tmp160 * _tmp290;
  const Scalar _tmp292 = _tmp163 * _tmp289;
  const Scalar _tmp293 = _tmp157 * (_tmp278 - _tmp279 - _tmp280 - _tmp281) +
                         _tmp24 * _tmp291 - _tmp24 * _tmp292;
  const Scalar _tmp294 = _tmp271 + _tmp291 * _tmp41 - _tmp292 * _tmp41;
  const Scalar _tmp295 =
      _tmp157 * _tmp276 + _tmp291 * _tmp47 - _tmp292 * _tmp47;
  const Scalar _tmp296 =
      _tmp156 * _tmp290 + _tmp162 * _tmp282 - _tmp164 * _tmp282;
  const Scalar _tmp297 = _tmp157 * (-_tmp272 - _tmp273 + _tmp274 - _tmp275) +
                         _tmp166 * _tmp282 - _tmp167 * _tmp282;
  const Scalar _tmp298 = -_tmp169 * _tmp282 + _tmp170 * _tmp282 + _tmp271;

  // Output terms (4)
  if (res != nullptr) {
    Eigen::Matrix<Scalar, 6, 1>& _res = (*res);

    _res(0, 0) = _tmp36;
    _res(1, 0) = _tmp42;
    _res(2, 0) = _tmp48;
    _res(3, 0) = _tmp103;
    _res(4, 0) = _tmp117;
    _res(5, 0) = _tmp124;
  }

  if (jacobian != nullptr) {
    Eigen::Matrix<Scalar, 6, 12>& _jacobian = (*jacobian);

    _jacobian(0, 0) = _tmp165;
    _jacobian(1, 0) = _tmp168;
    _jacobian(2, 0) = _tmp171;
    _jacobian(3, 0) = _tmp195;
    _jacobian(4, 0) = _tmp201;
    _jacobian(5, 0) = _tmp202;
    _jacobian(0, 1) = _tmp212;
    _jacobian(1, 1) = _tmp213;
    _jacobian(2, 1) = _tmp214;
    _jacobian(3, 1) = _tmp229;
    _jacobian(4, 1) = _tmp234;
    _jacobian(5, 1) = _tmp235;
    _jacobian(0, 2) = _tmp241;
    _jacobian(1, 2) = _tmp242;
    _jacobian(2, 2) = _tmp243;
    _jacobian(3, 2) = _tmp255;
    _jacobian(4, 2) = _tmp260;
    _jacobian(5, 2) = _tmp261;
    _jacobian(0, 3) = 0;
    _jacobian(1, 3) = 0;
    _jacobian(2, 3) = 0;
    _jacobian(3, 3) = _tmp262;
    _jacobian(4, 3) = _tmp263;
    _jacobian(5, 3) = _tmp264;
    _jacobian(0, 4) = 0;
    _jacobian(1, 4) = 0;
    _jacobian(2, 4) = 0;
    _jacobian(3, 4) = _tmp265;
    _jacobian(4, 4) = _tmp266;
    _jacobian(5, 4) = _tmp267;
    _jacobian(0, 5) = 0;
    _jacobian(1, 5) = 0;
    _jacobian(2, 5) = 0;
    _jacobian(3, 5) = _tmp268;
    _jacobian(4, 5) = _tmp269;
    _jacobian(5, 5) = _tmp270;
    _jacobian(0, 6) = _tmp277;
    _jacobian(1, 6) = _tmp283;
    _jacobian(2, 6) = _tmp288;
    _jacobian(3, 6) = 0;
    _jacobian(4, 6) = 0;
    _jacobian(5, 6) = 0;
    _jacobian(0, 7) = _tmp293;
    _jacobian(1, 7) = _tmp294;
    _jacobian(2, 7) = _tmp295;
    _jacobian(3, 7) = 0;
    _jacobian(4, 7) = 0;
    _jacobian(5, 7) = 0;
    _jacobian(0, 8) = _tmp296;
    _jacobian(1, 8) = _tmp297;
    _jacobian(2, 8) = _tmp298;
    _jacobian(3, 8) = 0;
    _jacobian(4, 8) = 0;
    _jacobian(5, 8) = 0;
    _jacobian(0, 9) = 0;
    _jacobian(1, 9) = 0;
    _jacobian(2, 9) = 0;
    _jacobian(3, 9) = _tmp59;
    _jacobian(4, 9) = _tmp104;
    _jacobian(5, 9) = _tmp119;
    _jacobian(0, 10) = 0;
    _jacobian(1, 10) = 0;
    _jacobian(2, 10) = 0;
    _jacobian(3, 10) = _tmp53;
    _jacobian(4, 10) = _tmp110;
    _jacobian(5, 10) = _tmp118;
    _jacobian(0, 11) = 0;
    _jacobian(1, 11) = 0;
    _jacobian(2, 11) = 0;
    _jacobian(3, 11) = _tmp56;
    _jacobian(4, 11) = _tmp108;
    _jacobian(5, 11) = _tmp120;
  }

  if (hessian != nullptr) {
    Eigen::Matrix<Scalar, 12, 12>& _hessian = (*hessian);

    _hessian.setZero();

    _hessian(0, 0) =
        std::pow(_tmp165, Scalar(2)) + std::pow(_tmp168, Scalar(2)) +
        std::pow(_tmp171, Scalar(2)) + std::pow(_tmp195, Scalar(2)) +
        std::pow(_tmp201, Scalar(2)) + std::pow(_tmp202, Scalar(2));
    _hessian(1, 0) = _tmp165 * _tmp212 + _tmp168 * _tmp213 + _tmp171 * _tmp214 +
                     _tmp195 * _tmp229 + _tmp201 * _tmp234 + _tmp202 * _tmp235;
    _hessian(2, 0) = _tmp165 * _tmp241 + _tmp168 * _tmp242 + _tmp171 * _tmp243 +
                     _tmp195 * _tmp255 + _tmp201 * _tmp260 + _tmp202 * _tmp261;
    _hessian(3, 0) = _tmp195 * _tmp262 + _tmp201 * _tmp263 + _tmp202 * _tmp264;
    _hessian(4, 0) = _tmp195 * _tmp265 + _tmp201 * _tmp266 + _tmp202 * _tmp267;
    _hessian(5, 0) = _tmp195 * _tmp268 + _tmp201 * _tmp269 + _tmp202 * _tmp270;
    _hessian(6, 0) = _tmp165 * _tmp277 + _tmp168 * _tmp283 + _tmp171 * _tmp288;
    _hessian(7, 0) = _tmp165 * _tmp293 + _tmp168 * _tmp294 + _tmp171 * _tmp295;
    _hessian(8, 0) = _tmp165 * _tmp296 + _tmp168 * _tmp297 + _tmp171 * _tmp298;
    _hessian(9, 0) = _tmp104 * _tmp201 + _tmp119 * _tmp202 + _tmp195 * _tmp59;
    _hessian(10, 0) = _tmp110 * _tmp201 + _tmp118 * _tmp202 + _tmp195 * _tmp53;
    _hessian(11, 0) = _tmp108 * _tmp201 + _tmp120 * _tmp202 + _tmp195 * _tmp56;
    _hessian(1, 1) =
        std::pow(_tmp212, Scalar(2)) + std::pow(_tmp213, Scalar(2)) +
        std::pow(_tmp214, Scalar(2)) + std::pow(_tmp229, Scalar(2)) +
        std::pow(_tmp234, Scalar(2)) + std::pow(_tmp235, Scalar(2));
    _hessian(2, 1) = _tmp212 * _tmp241 + _tmp213 * _tmp242 + _tmp214 * _tmp243 +
                     _tmp229 * _tmp255 + _tmp234 * _tmp260 + _tmp235 * _tmp261;
    _hessian(3, 1) = _tmp229 * _tmp262 + _tmp234 * _tmp263 + _tmp235 * _tmp264;
    _hessian(4, 1) = _tmp229 * _tmp265 + _tmp234 * _tmp266 + _tmp235 * _tmp267;
    _hessian(5, 1) = _tmp229 * _tmp268 + _tmp234 * _tmp269 + _tmp235 * _tmp270;
    _hessian(6, 1) = _tmp212 * _tmp277 + _tmp213 * _tmp283 + _tmp214 * _tmp288;
    _hessian(7, 1) = _tmp212 * _tmp293 + _tmp213 * _tmp294 + _tmp214 * _tmp295;
    _hessian(8, 1) = _tmp212 * _tmp296 + _tmp213 * _tmp297 + _tmp214 * _tmp298;
    _hessian(9, 1) = _tmp104 * _tmp234 + _tmp119 * _tmp235 + _tmp229 * _tmp59;
    _hessian(10, 1) = _tmp110 * _tmp234 + _tmp118 * _tmp235 + _tmp229 * _tmp53;
    _hessian(11, 1) = _tmp108 * _tmp234 + _tmp120 * _tmp235 + _tmp229 * _tmp56;
    _hessian(2, 2) =
        std::pow(_tmp241, Scalar(2)) + std::pow(_tmp242, Scalar(2)) +
        std::pow(_tmp243, Scalar(2)) + std::pow(_tmp255, Scalar(2)) +
        std::pow(_tmp260, Scalar(2)) + std::pow(_tmp261, Scalar(2));
    _hessian(3, 2) = _tmp255 * _tmp262 + _tmp260 * _tmp263 + _tmp261 * _tmp264;
    _hessian(4, 2) = _tmp255 * _tmp265 + _tmp260 * _tmp266 + _tmp261 * _tmp267;
    _hessian(5, 2) = _tmp255 * _tmp268 + _tmp260 * _tmp269 + _tmp261 * _tmp270;
    _hessian(6, 2) = _tmp241 * _tmp277 + _tmp242 * _tmp283 + _tmp243 * _tmp288;
    _hessian(7, 2) = _tmp241 * _tmp293 + _tmp242 * _tmp294 + _tmp243 * _tmp295;
    _hessian(8, 2) = _tmp241 * _tmp296 + _tmp242 * _tmp297 + _tmp243 * _tmp298;
    _hessian(9, 2) = _tmp104 * _tmp260 + _tmp119 * _tmp261 + _tmp255 * _tmp59;
    _hessian(10, 2) = _tmp110 * _tmp260 + _tmp118 * _tmp261 + _tmp255 * _tmp53;
    _hessian(11, 2) = _tmp108 * _tmp260 + _tmp120 * _tmp261 + _tmp255 * _tmp56;
    _hessian(3, 3) = std::pow(_tmp262, Scalar(2)) +
                     std::pow(_tmp263, Scalar(2)) +
                     std::pow(_tmp264, Scalar(2));
    _hessian(4, 3) = _tmp262 * _tmp265 + _tmp263 * _tmp266 + _tmp264 * _tmp267;
    _hessian(5, 3) = _tmp262 * _tmp268 + _tmp263 * _tmp269 + _tmp264 * _tmp270;
    _hessian(9, 3) = _tmp104 * _tmp263 + _tmp119 * _tmp264 + _tmp262 * _tmp59;
    _hessian(10, 3) = _tmp110 * _tmp263 + _tmp118 * _tmp264 + _tmp262 * _tmp53;
    _hessian(11, 3) = _tmp108 * _tmp263 + _tmp120 * _tmp264 + _tmp262 * _tmp56;
    _hessian(4, 4) = std::pow(_tmp265, Scalar(2)) +
                     std::pow(_tmp266, Scalar(2)) +
                     std::pow(_tmp267, Scalar(2));
    _hessian(5, 4) = _tmp265 * _tmp268 + _tmp266 * _tmp269 + _tmp267 * _tmp270;
    _hessian(9, 4) = _tmp104 * _tmp266 + _tmp119 * _tmp267 + _tmp265 * _tmp59;
    _hessian(10, 4) = _tmp110 * _tmp266 + _tmp118 * _tmp267 + _tmp265 * _tmp53;
    _hessian(11, 4) = _tmp108 * _tmp266 + _tmp120 * _tmp267 + _tmp265 * _tmp56;
    _hessian(5, 5) = std::pow(_tmp268, Scalar(2)) +
                     std::pow(_tmp269, Scalar(2)) +
                     std::pow(_tmp270, Scalar(2));
    _hessian(9, 5) = _tmp104 * _tmp269 + _tmp119 * _tmp270 + _tmp268 * _tmp59;
    _hessian(10, 5) = _tmp110 * _tmp269 + _tmp118 * _tmp270 + _tmp268 * _tmp53;
    _hessian(11, 5) = _tmp108 * _tmp269 + _tmp120 * _tmp270 + _tmp268 * _tmp56;
    _hessian(6, 6) = std::pow(_tmp277, Scalar(2)) +
                     std::pow(_tmp283, Scalar(2)) +
                     std::pow(_tmp288, Scalar(2));
    _hessian(7, 6) = _tmp277 * _tmp293 + _tmp283 * _tmp294 + _tmp288 * _tmp295;
    _hessian(8, 6) = _tmp277 * _tmp296 + _tmp283 * _tmp297 + _tmp288 * _tmp298;
    _hessian(7, 7) = std::pow(_tmp293, Scalar(2)) +
                     std::pow(_tmp294, Scalar(2)) +
                     std::pow(_tmp295, Scalar(2));
    _hessian(8, 7) = _tmp293 * _tmp296 + _tmp294 * _tmp297 + _tmp295 * _tmp298;
    _hessian(8, 8) = std::pow(_tmp296, Scalar(2)) +
                     std::pow(_tmp297, Scalar(2)) +
                     std::pow(_tmp298, Scalar(2));
    _hessian(9, 9) = std::pow(_tmp104, Scalar(2)) +
                     std::pow(_tmp119, Scalar(2)) + std::pow(_tmp59, Scalar(2));
    _hessian(10, 9) = _tmp104 * _tmp110 + _tmp118 * _tmp119 + _tmp53 * _tmp59;
    _hessian(11, 9) = _tmp104 * _tmp108 + _tmp119 * _tmp120 + _tmp56 * _tmp59;
    _hessian(10, 10) = std::pow(_tmp110, Scalar(2)) +
                       std::pow(_tmp118, Scalar(2)) +
                       std::pow(_tmp53, Scalar(2));
    _hessian(11, 10) = _tmp108 * _tmp110 + _tmp118 * _tmp120 + _tmp53 * _tmp56;
    _hessian(11, 11) = std::pow(_tmp108, Scalar(2)) +
                       std::pow(_tmp120, Scalar(2)) +
                       std::pow(_tmp56, Scalar(2));
  }

  if (rhs != nullptr) {
    Eigen::Matrix<Scalar, 12, 1>& _rhs = (*rhs);

    _rhs(0, 0) = _tmp103 * _tmp195 + _tmp117 * _tmp201 + _tmp124 * _tmp202 +
                 _tmp165 * _tmp36 + _tmp168 * _tmp42 + _tmp171 * _tmp48;
    _rhs(1, 0) = _tmp103 * _tmp229 + _tmp117 * _tmp234 + _tmp124 * _tmp235 +
                 _tmp212 * _tmp36 + _tmp213 * _tmp42 + _tmp214 * _tmp48;
    _rhs(2, 0) = _tmp103 * _tmp255 + _tmp117 * _tmp260 + _tmp124 * _tmp261 +
                 _tmp241 * _tmp36 + _tmp242 * _tmp42 + _tmp243 * _tmp48;
    _rhs(3, 0) = _tmp103 * _tmp262 + _tmp117 * _tmp263 + _tmp124 * _tmp264;
    _rhs(4, 0) = _tmp103 * _tmp265 + _tmp117 * _tmp266 + _tmp124 * _tmp267;
    _rhs(5, 0) = _tmp103 * _tmp268 + _tmp117 * _tmp269 + _tmp124 * _tmp270;
    _rhs(6, 0) = _tmp277 * _tmp36 + _tmp283 * _tmp42 + _tmp288 * _tmp48;
    _rhs(7, 0) = _tmp293 * _tmp36 + _tmp294 * _tmp42 + _tmp295 * _tmp48;
    _rhs(8, 0) = _tmp296 * _tmp36 + _tmp297 * _tmp42 + _tmp298 * _tmp48;
    _rhs(9, 0) = _tmp103 * _tmp59 + _tmp104 * _tmp117 + _tmp119 * _tmp124;
    _rhs(10, 0) = _tmp103 * _tmp53 + _tmp110 * _tmp117 + _tmp118 * _tmp124;
    _rhs(11, 0) = _tmp103 * _tmp56 + _tmp108 * _tmp117 + _tmp120 * _tmp124;
  }
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace circle_pose_graph
