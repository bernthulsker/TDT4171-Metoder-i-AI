T = [0.7 0.3;0.3 0.7];
o = [0.9 0.2];

O_1 = diag(o); % For rain
O_2 = diag(-(o-1)); %For no rain

evidence_1 = [1 1];
%if element i of => umbrella that day.

n = 2; %Setting number of days
F = zeros(2,n); %Inilializing data
F(:,1) = o';

for i = 2:n
    if evidence_1(i-1) %Look if umbrella
        F(:,i) = forward(O_1,T,F(:,i-1));
    else
        F(:,i) = forward(O_2,T,F(:,i-1));
    end
    
end
fprintf('It will rain on day two with p = %.3f\n',F(1,2));

evidence_2 = [1 1 0 1 1];
n = 5; 
F_2 = zeros(2,n);
F_2(:,1) = o';
for i = 2:n
    if evidence_2(i-1)
        F_2(:,i) = forward(O_1,T,F_2(:,i-1));
    else
        F_2(:,i) = forward(O_2,T,F_2(:,i-1));
    end
end
fprintf('The whole shabang:\n')
disp(F_2)
