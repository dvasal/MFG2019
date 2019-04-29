function V0  = reward2go_0_inf( g1v,g_n,u)

f = @(x) interp1(g1v,u,x) ;

V0 = f(g_n);


end
