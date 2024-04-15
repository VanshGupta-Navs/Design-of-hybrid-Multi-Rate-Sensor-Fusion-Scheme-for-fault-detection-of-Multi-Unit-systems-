function [xdot] = cstr_model(t, X)
    cstr_global; % <---- specific global variables ---->
    
    % global Ws Xs Us % steady state values of inputs
    global Wc u1 u2 % current values of inputs
    
    Ca = X(1);
    T = X(2);

    Fc = u2;
    F = u1;

    Cao = Wc(1); % Unmeasured disturbance
    G_Tcin = Wc(2);
    
    % ------- terms used for simplicity of writing the equation -----
    K = G_Ko * exp(-G_E / T);
    Gc = G_a / (G_V * G_rho * G_Cp);
    B1 = G_a * Fc^G_b / (2 * G_rhoc * G_Cpc);
    B = Fc^(G_b + 1) / (Fc + B1);
    r = G_delH / (G_rho * G_Cp);
    % --------------------------------------------------------------
    
    dCa_by_dt = F * (Cao - Ca) / G_V - K * Ca;
    
    dT_by_dt = F * (G_To - T) / G_V;
    dT_by_dt = dT_by_dt - Gc * B * (T - G_Tcin);
    dT_by_dt = dT_by_dt + r * K * Ca;
    
    xdot = [dCa_by_dt; dT_by_dt];
end

