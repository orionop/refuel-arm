function [Q, is_LS] = KR6_R700(R, T)
    [Q, is_LS] = IK.IK_spherical_2_parallel(R, T, hardcoded_IK_setups.KR6_R700.get_kin());
end

