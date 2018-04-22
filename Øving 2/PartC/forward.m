function f_tp1 = forward(O,T,f)
h = O*T'*f; %Dummy
f_tp1 = h.*(1/sum(h)); %Normalize by deviding by sum of elements