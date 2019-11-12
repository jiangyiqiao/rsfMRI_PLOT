
p = rand(1,422);
count = 0
nc = datas(10);
emci = datas(100);
lmci = datas(200);
for i=[1:1:422]
    test_data = [nc(2:130,i),average_emci(2:130,i),average_lmci(2:130,i)];
    pp = friedman(test_data,1,'off');
    if pp<0.05
        count =count + 1;
        disp(i);
    end
    p(i)=pp;
    
end