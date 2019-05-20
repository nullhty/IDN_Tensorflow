clear; close all;
%% settings
dataDir = '291/';  % 291 augment dataset
savepath = 'train_hdf5/train_x2.h5';  % save filename
size_input = 29; % 29 | 39
size_label = 58; % 57 | 77
up_scale = 2; % upsacling factor
stride_label = 58
%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);

count_input = 0;
count_label = 0;
count_bic = 0;

%% generate data
%% filepaths=dir(fullfile(folder,'*.bmp'));
f_lst = [];
f_lst = [f_lst; dir(fullfile(dataDir, '*.jpg'))];
f_lst = [f_lst; dir(fullfile(dataDir, '*.bmp'))];
for f_iter = 1:numel(f_lst)
    f_iter
	f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end
    f_path = fullfile(dataDir,f_info.name);
    image0 = imread(f_path);
    if size(image0,3) == 1
        continue;
    end
    for angle = 0: 90 :270
		im_rot = imrotate(image0, angle);
		
		for scale = 1.0 : -0.1 : 0.6
			im_down = imresize(im_rot, scale, 'bicubic');
			
			for j = 3 : -2 : 1  % 3--> not flip, 1-->flip horizontally
                if j == 3
                    image = im_down;
                else
                    image = flip(im_down, j);
                end
				image = rgb2ycbcr(image);
				image = im2double(image(:, :, 1)); % uint8 to double, ranges from [16/255, 235/255]

				im_label = modcrop(image, up_scale); % high resolution subimage
				[hei_label,wid_label]=size(im_label);% HR subimage size
				
				for x = 1 : stride_label : hei_label -size_label + 1
					for y = 1 : stride_label : wid_label -size_label +1
						
						sub_label=im_label(x:x+size_label-1,y:y+size_label-1);
						sub_input = imresize(sub_label, [size_input,size_input], 'bicubic');
						count_label=count_label+1;
						label(:, :, 1, count_label)=single(sub_label);
						data(:, :, 1, count_label)=single(sub_input);
					end
				end
			end
		end
    end
end

order = randperm(count_label);
data = data(:, :, 1, order);
label = label(:, :, 1, order);


%data=permute(data,[4 3 1 2]);
%label=permute(label,[4 3 1 2]);

h5create('train_IDN.h5','/data',size(data),'Datatype','single');
h5create('train_IDN.h5','/label',size(label),'Datatype','single');

h5write('train_IDN.h5','/data',data);
h5write('train_IDN.h5','/label',label);