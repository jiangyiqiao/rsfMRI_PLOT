load('data/F_test/ave_nc.mat');
load('data/F_test/ave_emci.mat');
load('data/F_test/ave_lmci.mat');

p = rand(1,422);
for i=[1:1:422]
    
    [h,pp,ci] = ttest(average_nc(:,i),average_emci(:,i));
    if pp<0.05
        disp(i);
    end
    p(i)=pp;
    
end