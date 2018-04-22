function [sv,f,b] = forBackWard(ev,prior)
    t = length(ev);
    f = zeros(length(prior),t);
    
    sv = zeros(length(prior),t);
    b = ones(length(prior),1);
    
    T = [0.7 0.3;0.3 0.7];
    o = [0.9 0.2];

    O_1 = diag(o); % For rain
    O_2 = diag(-(o-1)); %For no rain
    
    f(:,1) = prior';

    for i = 2:t
        if ev(i)
            f(:,i) = forward(O_1,T,f(:,i-1));
        else
            f(:,i) = forward(O_2,T,f(:,i-1));
        end
    end
    
    for j = 0:t-1
        i = t-j;
        sv = (f(:,i).*b).* (1/(sum((f(:,i).*b))));
        if ev(i)
            b = backward(O_1,T,b);
        else
            b = backward(O_2,T,b);
        end
        fprintf('b_%d = \n',i)
        disp(b)
    end
end

