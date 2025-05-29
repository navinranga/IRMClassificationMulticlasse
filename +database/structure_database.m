%-- This function is used to 
function [database] = structure_database(img1,img2,img3,img4,img5,img6,img7,img8,img9,nclasse)

    %size of each group :
    Train = 80;
    Valid = 10;
    Test = 10;
    
    img = [img1;img2;img3;img4;img5;img6;img7;img8;img9];
    
    database.X_train = []; database.X_valid = []; database.X_test = [];
    database.Y_train = []; database.Y_valid = []; database.Y_test = [];

    for l=1:nclasse

        nbImg = size(img(l,:),2);
        orderImg = randperm(nbImg);

        img_train = []; img_valid = []; img_test = [];
        result_train = []; result_valid = []; result_test = [];

        imgx = img(l,:);
        
        resultx = zeros(nclasse,1); resultx(l) = 1;

        for i=1:Train
            img_train = [img_train, imgx{orderImg(i)}(:)];
            result_train = [result_train, resultx];
        end

        for e=(i+1):(i+Valid)
            img_valid = [img_valid, imgx{orderImg(e)}(:)];
            result_valid = [result_valid, resultx];
        end
        
        for f=(e+1):(e+Test)
            img_test = [img_test, imgx{orderImg(f)}(:)];
            result_test = [result_test, resultx];
        end

        database.X_train = [database.X_train, img_train]; database.X_valid = [database.X_valid, img_valid]; database.X_test = [database.X_test, img_test];
        database.Y_train = [database.Y_train, result_train]; database.Y_valid= [database.Y_valid, result_valid]; database.Y_test = [database.Y_test, result_test];

    end

end

