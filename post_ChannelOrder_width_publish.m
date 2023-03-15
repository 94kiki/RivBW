%%
% This code is used for the post-process of DeepLabV3+, get the main channel
% and tributary channel 1. order the channel 2. use width algorithm to get the
% channel width automatic by Wenqi, 2022-10-06
% add connective.m to the path
% this copy used to process Nicola river data
% current version used to process Enrico's data, Feb 7, 2023
clc,clear;close all;
ts=tic;

boundary_left=table2array(readtable('/Users/kiki-mac/Desktop/NicolaRiver/v6/valley_left2.csv'));
boundary_right=table2array(readtable('/Users/kiki-mac/Desktop/NicolaRiver/v6/valley_right2.csv'));

boundary_left_sort=path_sort_fun(1:size(boundary_left,1),boundary_left(1,:),boundary_left);
boundary_right_sort=path_sort_fun(1:size(boundary_right,1),boundary_right(1,:),boundary_right);

boundary_left=boundary_left(boundary_left_sort,:);
boundary_right=boundary_right(boundary_right_sort,:);

boundary1=unique(boundary_left,'rows','stable');
boundary2=unique(boundary_right,'rows','stable');
boundary1(isnan(boundary1(:,1)),:)=[];
boundary2(isnan(boundary2(:,1)),:)=[];

boundary1=interparc(10000,boundary1(:,1),boundary1(:,2),'lin');
boundary2=interparc(10000,boundary2(:,1),boundary2(:,2),'lin');

boundary.x=ceil([boundary1(:,1);fliplr(boundary2(:,1)')';boundary1(1,1)]);
boundary.y=ceil([boundary1(:,2);fliplr(boundary2(:,2)')';boundary1(1,2)]);

island_bdy=readtable('/Users/kiki-mac/Desktop/NicolaRiver/v6/islands.xlsx','Sheet','2021');
island_bdy=[island_bdy.id,island_bdy.x,island_bdy.y];

island_id=unique(island_bdy(:,1));

for ii=1:length(island_id)
    loc=find(island_bdy(:,1)==island_id(ii));
    B_island{ii,1}=island_bdy(loc,2:3);
end

mainchannel_boundary=[boundary.x,boundary.y];
spacing=1;% set the spacing of transects
writetbl=channel_width_order(mainchannel_boundary,B_island,spacing);
writetable(writetbl,'/Users/kiki-mac/Desktop/NicolaRiver/v6/channel_width_2021.xlsx')

%%
function writetbl=channel_width_order(mainchannel_boundary,B_island,spacing)
river_path_forward=[];
river_path_backward=[];

%% get the island boundary
ts=tic;
island=zeros(length(B_island),1);

tic
parfor k = 1:length(B_island)
    boundary = B_island{k};
    in=inpolygon(boundary(:,1), boundary(:,2),mainchannel_boundary(:,1),mainchannel_boundary(:,2));

    if sum(in)~=0
        island(k)=k;
    end
end
fprintf('The time to get Island Boundary: %.2f seconds.\n',toc )
island(island==0)=[];% get the island in the river boundary
num_pt=size(mainchannel_boundary,1);
raw_main_bdy=mainchannel_boundary;
if num_pt*10<=10000
    mainchannel_boundary=interparc(num_pt*10,mainchannel_boundary(:,1),mainchannel_boundary(:,2),'lin');
else
    mainchannel_boundary=interparc(10000,mainchannel_boundary(:,1),mainchannel_boundary(:,2),'lin');
end
%% voronoi polygon
% show all connections
disp('Creating Voronoi polygons... ')

[vx,vy]=voronoi(mainchannel_boundary(:,1),mainchannel_boundary(:,2));
vvx=reshape(vx,[],1);
vvy=reshape(vy,[],1);
vxy=[vvx,vvy];
[vxx,~,ic]=unique(vxy,'rows');
presentTimes = accumarray(ic,1);
voronoi_points=[vxx,presentTimes];

% remove the voronoi points less than 3 connections and negative coordinates
voronoi_points(voronoi_points(:,1)<=0|voronoi_points(:,2)<=0|voronoi_points(:,3)<3,:)=[];

tic
iii=zeros(size(voronoi_points,1),1);
main_bdy_x=raw_main_bdy(:,1);
main_bdy_y=raw_main_bdy(:,2);
voronoi_points_x=voronoi_points(:,1);
voronoi_points_y=voronoi_points(:,2);
% if GPU can be used, then use GPU
if gpuDeviceCount("available")>=1
    iii=inpolygon(gpuArray(voronoi_points_x),gpuArray(voronoi_points_y),gpuArray(main_bdy_x),gpuArray(main_bdy_y));
    iii=gather(iii);
else
    parfor kk=1:size(voronoi_points,1)
        in=inpolygon(voronoi_points_x(kk),voronoi_points_y(kk),main_bdy_x,main_bdy_y);
        if in==1
            iii(kk)=1;
        end
    end
end
iii=logical(iii);
fprintf('The time to get inpolygon points: %.2f seconds.\n',toc )

voronoi_points_in_channel=array2table(voronoi_points(iii,:),'VariableNames',{'x';'y';'connections'});

%% post process the voronoi diagram
vp=[voronoi_points_in_channel.x,voronoi_points_in_channel.y];
tic
disp('Creating the Connective Martix...')
[K,N]=connective(vp,vx,vy); 

N_repeat_forward=N;
fprintf('The time to get connective martix: %.2f seconds.\n',toc )
%% step 1, forward, decrease the N martix, to remove outliners and get the main centreline pints
% export forward_mark for the next use
tic
forward_steps=300;
decrease_range=zeros(size(N,2),forward_steps);
decrease_number=zeros(forward_steps,4);
scan_mark=zeros(size(N,2),1);
f=1;
while f<=300
    connection_temp=sum(N,1)';
    N(connection_temp==1,:)=0;
    N(:,connection_temp==1)=0;
    decrease_range(:,f)=connection_temp;% remain connection

    sum_temp=sum(N,1)';
    decrease_number(f,:)=[f,sum(sum_temp==1),sum(sum_temp==2),sum(sum_temp==3)];%
    scan_mark(connection_temp==1,:)=f;% record when the point deleted

    if sum(sum_temp==1)==2% if only two vertices, stop the searching
        break
    end
    
    f=f+1;
    fprintf('Current search loop: %d \n',f);
end
% end
fprintf('The time to get forward path: %.2f seconds.\n',toc )

% toc
decrease_number(decrease_number(:,1)==0,:)=[];
decrease_number(:,5)=movmean(gradient(decrease_number(:,2)),100);% add the movmean gradient of each column data, 100 is the movmean window
decrease_number(:,6)=movmean(gradient(decrease_number(:,3)),100);
decrease_number(:,7)=movmean(gradient(decrease_number(:,4)),100);

if length(unique(decrease_number(:,7)))~=1
    scan_choose=min(decrease_number(abs(decrease_number(:,7))<=0.05,1));% get f of the movmean gradient<=0.05
else
    scan_choose=max(decrease_number(:,1))+1;
end
% get the final connection martix
if ~isempty(scan_choose)
    N_repeat_forward(scan_mark>0&scan_mark<scan_choose,:)=0;
    N_repeat_forward(:,scan_mark>0&scan_mark<scan_choose)=0;
else
    scan_choose=max(decrease_number(:,1))+1;
    N_repeat_forward(scan_mark>0&scan_mark<scan_choose,:)=0;
    N_repeat_forward(:,scan_mark>0&scan_mark<scan_choose)=0;
end
connection_temp=sum(N_repeat_forward,2);
connection_1=find(connection_temp==1);% nodes with 1 connection
connection_3=find(connection_temp==3);% nodes with 3 connections, all index in vp
[K_new_rows,K_new_cols]=find(triu(N_repeat_forward));% export conneciton martix without repeat points
K_new_forward=sortrows([K_new_rows,K_new_cols],1); % K_new used for forward search
K_new_forward_backup=K_new_forward;
K_new_backward=K;
% show the final figure of this step
connection_final=sum(N_repeat_forward,2);

%% step 2, backward, expand river path to the inlet, to get the branches
% start from connection_1, and according to the remove record information
% to back ward,
% stop at the vertices with no next connections or the node with two connections
% which these two connections forward_mark larger than one,
% get back the nodes removed before
tic
if length(connection_1)~=2
    river_path_backward=zeros(size(N,1),length(connection_1));
    river_path_backward(1,:)=connection_1';
    for cc=1:length(connection_1)
        ii=2;
        current_search=connection_1(cc);

        next_point=findNextPoint_backward(current_search,K_new_backward,scan_mark);
        if isempty(next_point)
            continue;
        else
            river_path_backward(ii,cc)=next_point;

        end

        while (length(next_point)<2 && ~isempty(next_point) && scan_mark(next_point)~=1) ||(length(next_point)==2 && min(scan_mark(next_point))==1)
            if min(scan_mark(next_point))==1 && length(next_point)>1
                if sum(scan_mark(next_point)==1)==2 % two 1 connection vertices, just choose one vertice
                    next_point(1)=[];
                else
                    next_point(scan_mark(next_point)==1)=[];
                end
            end
            river_path_backward(ii,cc)=next_point;
            current_search=next_point;
            f=parfeval(backgroundPool,@findNextPoint_backward,1,current_search,K_new_backward,scan_mark);
            next_point=fetchOutputs(f);
            ii=ii+1;

            fprintf('Backward processing, cc: %d and ii: %d \n',cc,ii)
        end
    end
    river_path_backward(sum(river_path_backward,2)==0,:)=[];
    fprintf('The time to get backward path: %.2f seconds.\n',toc )


else
    disp('No backward path identified!!!')
end



%% step 3, get the tributary
% start the conneciton_1 nodes, forward to get the river path2,
% stop at the node with 3 connections (appeared in connection_3)
tic;
river_path_forward=zeros(size(N,1),length(connection_1));
vp_num=size(vp,1);
for cc_1=1:length(connection_1)
    K_new_forward=K_new_forward_backup;
    river_path_forward(1,cc_1)=connection_1(cc_1);
    ii=2;
    current_search=connection_1(cc_1);
    f=parfeval(backgroundPool,@findNextPoint_forward,1,current_search,K_new_forward,scan_mark,ii,cc_1,river_path_forward);
    next_point=fetchOutputs(f);
    if isempty(next_point) || ismember(next_point,setdiff(river_path_forward,connection_1))
        continue;
    end

    loc=ismember(K_new_forward,[current_search,next_point],'rows')|ismember(K_new_forward,[next_point,current_search],'rows');
    K_new_forward(loc,:)=[];

    river_path_forward(ii,cc_1)=next_point;
    if length(connection_1)~=2
        end_point=connection_3;
    else
        end_point=connection_1;
    end
    while ~ismember(next_point,end_point)  && ii <=vp_num
        current_search=next_point;
        f=parfeval(backgroundPool,@findNextPoint_forward,1,current_search,K_new_forward,scan_mark,ii,cc_1,river_path_forward);
        next_point=fetchOutputs(f);
        if isempty(next_point) || ismember(next_point,setdiff(river_path_forward,connection_1))
            break;% break, out of while loop to for loop still in the for loop, continue back to the while loop still in the while loop, return out of the whole function
        end

        ii=ii+1;
        river_path_forward(ii,cc_1)=next_point;
        loc=ismember(K_new_forward,[current_search,next_point],'rows')|ismember(K_new_forward,[next_point,current_search],'rows');
        K_new_forward(loc,:)=[];


        fprintf('Forward processing, cc: %d and ii: %d \n',cc_1,ii)
    end
end
river_path_forward(sum(river_path_forward,2)==0,:)=[];
fprintf('The time to get tributary path: %.2f seconds.\n',toc )


if length(connection_1)==2
    river_path_forward=river_path_forward(:,1);
end




%% step 4, get the path of the order 2 path segments (connection==2),
% start from connection_3, stop at the point with only 1 connection or 2 connections(not good, will go a wrong way)

% path_order2, start from connection_1, stop at the point of connections_3
connections2_idx=find(connection_final==2);% 2 connections
connections3_idx=find(connection_final==3);% 3 connections

if length(connections3_idx)>1

    part_tributies=river_path_forward(:);
    % part_tributies=part_tributies(:);
    part_tributies(part_tributies==0)=[];
    left_points=setdiff(connections2_idx,part_tributies);% extract the points connect with connection_3, index of vp
    left_points=[left_points;connections3_idx];
    left_points=sort(left_points);

    [K_left,N_left]=connective(vp(left_points,:),vx,vy);
    % scatter(vp(left_points,1),vp(left_points,2),50,"k",'filled')
    sum_temp_left=sum(N_left,2);
    left_connection3=left_points(sum_temp_left==3);
    left_connection1=left_points(sum_temp_left==1);
    K_left_search=K_left;
    path_order2=zeros(length(left_points),length(left_connection1));
    for kkk=1:length(left_connection1)
        start_index=find(left_points==left_connection1(kkk));
        iteration=1;
        path_order2(iteration,kkk)=left_points(start_index);
        [nextpoint,~]=FindNextPoint(start_index,K_left_search);
        while ~isempty(nextpoint) && ~ismember(left_points(nextpoint),left_connection3)
            [nextpoint,K_left_search]=FindNextPoint(start_index,K_left_search);
            iteration=iteration+1;
            if isempty(nextpoint)
                break;
            end
            path_order2(iteration,kkk)=left_points(nextpoint);
            start_index=nextpoint;

            fprintf('Search Loop, path: %d, iteration: %d. \n',kkk,iteration)
            %         if iteration==624
            %             cc=0;
            %         end
        end
        if length(nonzeros(path_order2(:,kkk)))<=1
            path_order2(:,kkk)=[];
        end
    end
    path_order2(sum(path_order2,2)==0,:)=[];

    disp('path_order2 generated!!!')
else
    path_order2=[];
    disp('path_order2 not generated!!!')
end


%% get the path from the node to the vertics with 3connections, still left the path between 3connections

% step 5, get the path between 3connections
if ~isempty(path_order2)
    path_3connections=path_order2(:);
    path_3connections(path_3connections==0)=[];
    left_3connections_points=setdiff(left_points,path_3connections);
    left_3connections_points=[left_3connections_points;left_connection3];
    left_3connections_points=sort(left_3connections_points);

    numPoints=length(left_3connections_points);
    x=vp(left_3connections_points,1);
    y=vp(left_3connections_points,2);
    % according to the nearest distance to find next points, may not right,
    % still need to according to connection martix
    % Make a list of which points have been visited
    path_left_3connections=zeros(numPoints,length(left_connection3)-1);
    for kkk=1:length(left_connection3)-1
        currentIndex=find(left_3connections_points==left_connection3(kkk));
        left_connection3(kkk)=[];
        beenVisited = false(1, numPoints);
        % Make an array to store the order in which we visit the points.
        visitationOrder = zeros(1, numPoints);
        % Define a filasafe
        maxIterations = numPoints + 1;
        iterationCount = 1;
        % Visit each point, finding which unvisited point is closest.
        % Define a current index.  currentIndex will be 1 to start and then will vary.

        %     currentIndex = 1;
        while sum(beenVisited) < numPoints && iterationCount < maxIterations && ~isempty(currentIndex) && ~ismember(left_3connections_points(currentIndex),left_connection3)
            % Indicate current point has been visited.
            visitationOrder(iterationCount) = currentIndex;
            %         scatter(vp(left_points(currentIndex),1),vp(left_points(currentIndex),2),'g','filled')
            %         drawnow
            beenVisited(currentIndex) = true;
            % Get the x and y of the current point.
            thisX = x(currentIndex);
            thisY = y(currentIndex);
            %   text(thisX + 0.01, thisY, num2str(currentIndex), 'FontSize', 35, 'Color', 'r');
            % Compute distances to all other points
            distances = sqrt((thisX - x) .^ 2 + (thisY - y) .^ 2);
            % Don't consider visited points by setting their distance to infinity.
            distances(beenVisited) = inf;
            % Also don't want to consider the distance of a point to itself, which is 0 and would alsoways be the minimum distances of course.
            distances(currentIndex) = inf;
            % Find the closest point.  this will be our next point.
            %         [minDistance, indexOfClosest] = min(distances);
            [~, indexOfClosest] = min(distances);
            % Save this index
            iterationCount = iterationCount + 1;
            % Set the current index equal to the index of the closest point.
            currentIndex = indexOfClosest;

        end
        visitationOrder(visitationOrder==0)=[];
        %     path_sort=left_points(visitationOrder);
        path_left_3connections(1:length(visitationOrder),kkk)=left_3connections_points(visitationOrder);% index to vp
        path_left_3connections(end,kkk)=left_3connections_points(currentIndex);
        %     iterationCount = 1;
    end

else
    path_left_3connections=[];
    disp('path_left_3connections not generated!!!')
end






%% sort segments, tributry centerlines
disp('Sorting segment centerline...')
tic
if exist('river_path_backward','var')==0
    river_path_backward=[];
end
path_sort_tributary=zeros(size([river_path_forward;river_path_backward]));
for jjjj=1:size(river_path_forward,2)
    if ~isempty(river_path_backward)
        river_path_tributary=[river_path_forward(:,jjjj);river_path_backward(2:end,jjjj)];
    else
        river_path_tributary=river_path_forward(:,jjjj);
    end
    river_path_tributary(river_path_tributary==0)=[];
    [~,N_center]=connective(vp(river_path_tributary,:),vx,vy);
    temp_center=sortrows(vp(river_path_tributary,:),1);
    certer_vertics=temp_center(sum(N_center,2)==1,:); % find the first or end point of center points

    temp_path=path_sort_fun(river_path_tributary,certer_vertics(1,:),vp);
    path_sort_tributary(1:length(temp_path),jjjj)=temp_path;

end
path_sort_tributary(sum(path_sort_tributary,2)==0,:)=[];
fprintf('Sorted centerline extracted, %.2f s!\n',toc)




%% use the cirle of connection3 points to search the nearest tributary mouth, to get the tributary boundary (order 1)
% order1 centerline and boundary
disp('Extracting the order1 centerline and bounday...')
tic
mainchannel_boundary2=mainchannel_boundary;
for pst=1:size(path_sort_tributary,2)
    test_tri=path_sort_tributary(:,pst);
    test_tri(test_tri==0)=[];
    num_digit=numel(num2str(size(path_sort_tributary,2)));

    [idx0,idx]=ismember(connection_3,test_tri);
    idx0=find(idx0);
    if length(connection_1)==2 || isempty(connection_3) || isempty(idx0)% if the left points only two and all belong to order3_connection3
        mainchannel_boundary2(end+1,:)=mainchannel_boundary2(1,:);
        data.data_bdy.order1.(['t',num2str(pst,['%0',num2str(num_digit+1),'.f'])])=mainchannel_boundary2;% save the tributary boundary to tributary.tb**
        data.data_ctl.order1.(['t',num2str(pst,['%0',num2str(num_digit+1),'.f'])])=vp(test_tri,:);% save the tributary centerline to tributary.tc**


        continue;
    end
    center=vp(connection_3(idx0),:);
    min_r=min(sqrt((center(1,1)-mainchannel_boundary(:,1)).^2+(center(1,2)-mainchannel_boundary(:,2)).^2));

    th = linspace(0,2*pi) ;

    for search=1:20 % search no more than 20 times
        cc_x = center(1)+min_r*cos(th) ;
        cc_y = center(2)+min_r*sin(th) ;
        % intersection of circle with boundary
        [c_b_x,c_b_y] = intersections(cc_x,cc_y,mainchannel_boundary(:,1),mainchannel_boundary(:,2),1);
        % intersection of circle with tributary centerline(tc)
        [c_tc_x,c_tc_y] = intersections(vp(test_tri,1),vp(test_tri,2),cc_x,cc_y,1);
        if isempty(c_tc_x)||isempty(c_b_x)
            min_r=min_r+1;
            continue;
        end
        k_c_tc=(center(2)-c_tc_y)/(center(1)-c_tc_x);
        b_c_tc=center(2)-k_c_tc*center(1);
        d_temp=k_c_tc*c_b_x+b_c_tc-c_b_y;% find the location of bdy intersection to the line of center to intersection of circle with tributary centerline(tc)
        d_temp1=find(d_temp<=0);
        d_temp2=find(d_temp>0);

        % get the angle of intersection of circle/tri_centerline to cirlce center
        th_c_tc=atan2d((c_tc_y-center(2)),(c_tc_x-center(1)));
        if th_c_tc<0
            th_c_tc=th_c_tc+360;
        end
        % get the angle of intersec of circle/boundary to circle center
        th_c_b=atan2d((c_b_y-center(2)),(c_b_x-center(1)));
        if th_c_tc>=0&&th_c_tc<90

        elseif th_c_tc>=90&&th_c_tc<270
            th_c_b(th_c_b<0)=th_c_b(th_c_b<0)+360;
        elseif th_c_tc>=270&&th_c_tc<360
            th_c_tc=th_c_tc-360;
        end
        if length(th_c_b)>=2 && ~isempty(th_c_b(th_c_b>=th_c_tc))&& ~isempty(th_c_b(th_c_b<th_c_tc))&&~isempty(d_temp1)&&~isempty(d_temp2)
            diff_th=th_c_tc-th_c_b;
            diff_th=[d_temp,diff_th];
            diff_th1=diff_th(d_temp1,:);
            diff_th2=diff_th(d_temp2,:);
            [~,min_th1]=min(abs(diff_th1(:,2)));
            [~,min_th2]=min(abs(diff_th2(:,2)));
            [~,min_th1_idx]=ismember(diff_th1(min_th1,:),diff_th,'rows');
            [~,min_th2_idx]=ismember(diff_th2(min_th2,:),diff_th,'rows');
            tri_mouth=[c_b_x([min_th1_idx;min_th2_idx]),c_b_y([min_th1_idx;min_th2_idx])];
            [~,tri_boundary1]=min(sqrt((tri_mouth(1,1)-mainchannel_boundary(:,1)).^2+...
                (tri_mouth(1,2)-mainchannel_boundary(:,2)).^2));
            [~,tri_boundary2]=min(sqrt((tri_mouth(2,1)-mainchannel_boundary(:,1)).^2+...
                (tri_mouth(2,2)-mainchannel_boundary(:,2)).^2));
            tri_boundary_idx=[tri_boundary1,tri_boundary2];
            tri_boundary_t=mainchannel_boundary(min(tri_boundary_idx):max(tri_boundary_idx),:);
            test_tri_incc=inpolygon(vp(test_tri,1),vp(test_tri,2),cc_x,cc_y);% centerline in the circle
            test_tri2=test_tri;%copy
            [itsec_x,itsec_y]=intersections(tri_mouth(:,1),tri_mouth(:,2),vp(test_tri,1),vp(test_tri,2),1);

            test_tri(test_tri_incc)=[];
            in=inpolygon(vp(test_tri,1),vp(test_tri,2),tri_boundary_t(:,1),tri_boundary_t(:,2));
            if sum(in)/length(test_tri)<=0.8 % if less than 80% centerline in the tributary polygon
                % need to note that this boundary include the the final segment, while others does not
                tri_boundary_t=setdiff(mainchannel_boundary,tri_boundary_t,'rows','stable');
            end
            in2=inpolygon(vp(test_tri2,1),vp(test_tri2,2),tri_boundary_t(:,1),tri_boundary_t(:,2));
            test_tri_f=test_tri2(in2);
            data.data_bdy.order1.(['t',num2str(pst,['%0',num2str(num_digit+1),'.f'])])=tri_boundary_t;% save the tributary boundary to tributary.tb**
            mainchannel_boundary2=setdiff(mainchannel_boundary2,tri_boundary_t,'rows','stable');
            path_sort=path_sort_fun([vp(test_tri_f,:);[itsec_x,itsec_y]],[itsec_x,itsec_y],vp);% output coordinates
            data.data_ctl.order1.(['t',num2str(pst,['%0',num2str(num_digit+1),'.f'])])=path_sort;% save the tributary centerline to tributary.tc**

            path_sort=[];
            break;
            %           end
        else
            min_r=min_r+1;
        end
    end

end
fprintf('The order1 centerline and bounday extracted, %.2f s!\n',toc)



%% use the cirle of connection3 points to search the nearest order 2 channel mouth, to get the order channel2 boundary (order 2)
% order2 centerline and boundary
disp('Extracting the order2 centerline and bounday...')
tic
if ~isempty(path_order2)
    for po=1:size(path_order2,2)

        temp_path=path_order2(:,po);
        temp_path(temp_path==0,:)=[];
        order2_connection3=left_points(sum_temp_left==3);
        if isempty(order2_connection3)
            order2_connection3=connections3_idx;

            if isempty(order2_connection3)

                break;% only left one path to search
            end
        end


        num_digit=numel(num2str(size(path_order2,2)));
        [idx0,~]=ismember(order2_connection3,temp_path);
        idx0=find(idx0);
        if length(idx0)==length(order2_connection3) % if the left points only two and all belong to order3_connection3
            mainchannel_boundary2(end+1,:)=mainchannel_boundary2(1,:);
            data.data_bdy.order2.(['t',num2str(po,['%0',num2str(num_digit+1),'.f'])])=mainchannel_boundary2;% save the tributary boundary to tributary.tb**
            data.data_ctl.order2.(['t',num2str(po,['%0',num2str(num_digit+1),'.f'])])=vp(temp_path,:);% save the tributary centerline to tributary.tc**

            continue;
        end

        center=vp(order2_connection3(idx0),:);
        min_r=min(sqrt((center(1,1)-mainchannel_boundary2(:,1)).^2+(center(1,2)-mainchannel_boundary2(:,2)).^2));
        th = linspace(0,2*pi) ;

        for search=1:20 % search no more than 20 times
            cc_x = center(1)+min_r*cos(th) ;
            cc_y = center(2)+min_r*sin(th) ;
            % intersection of circle with boundary
            [c_b_x,c_b_y] = intersections(cc_x,cc_y,mainchannel_boundary2(:,1),mainchannel_boundary2(:,2),1);
            % intersection of circle with tributary centerline(tc)
            [c_tc_x,c_tc_y] = intersections(vp(temp_path,1),vp(temp_path,2),cc_x,cc_y,1);
            if isempty(c_tc_x)||isempty(c_b_x)
                min_r=min_r+1;
                continue;
            end
            k_c_tc=(center(2)-c_tc_y)/(center(1)-c_tc_x);
            b_c_tc=center(2)-k_c_tc*center(1);
            d_temp=k_c_tc*c_b_x+b_c_tc-c_b_y;% find the location of bdy intersection to the line of center to intersection of circle with tributary centerline(tc)
            d_temp1=find(d_temp<=0);
            d_temp2=find(d_temp>0);
            % get the angle of intersection of circle/tri_centerline to cirlce center
            th_c_tc=atan2d((c_tc_y-center(2)),(c_tc_x-center(1)));
            if th_c_tc<0
                th_c_tc=th_c_tc+360;
            end
            % get the angle of intersec of circle/boundary to circle center
            th_c_b=atan2d((c_b_y-center(2)),(c_b_x-center(1)));
            if th_c_tc>=0&&th_c_tc<90

            elseif th_c_tc>=90&&th_c_tc<270
                th_c_b(th_c_b<0)=th_c_b(th_c_b<0)+360;
            elseif th_c_tc>=270&&th_c_tc<360
                th_c_tc=th_c_tc-360;
            end

            if length(th_c_b)>=2 && ~isempty(th_c_b(th_c_b>=th_c_tc))&& ~isempty(th_c_b(th_c_b<th_c_tc))&&~isempty(d_temp1)&&~isempty(d_temp2)
                diff_th=th_c_tc-th_c_b;
                diff_th=[d_temp,diff_th];
                diff_th1=diff_th(d_temp1,:);
                diff_th2=diff_th(d_temp2,:);
                [~,min_th1]=min(abs(diff_th1(:,2)));
                [~,min_th2]=min(abs(diff_th2(:,2)));
                [~,min_th1_idx]=ismember(diff_th1(min_th1,:),diff_th,'rows');
                [~,min_th2_idx]=ismember(diff_th2(min_th2,:),diff_th,'rows');
                % find the intersections between the centerline with the line of tri mouth
                intsec=intersections(c_b_x([min_th1_idx,min_th2_idx]),c_b_y([min_th1_idx,min_th2_idx]),vp(temp_path,1),vp(temp_path,2),1);
                if isempty(intsec)
                    min_r=min_r+1;
                else

                    tri_mouth=[c_b_x([min_th1_idx;min_th2_idx]),c_b_y([min_th1_idx;min_th2_idx])];
                    [~,tri_boundary1]=min(sqrt((tri_mouth(1,1)-mainchannel_boundary2(:,1)).^2+...
                        (tri_mouth(1,2)-mainchannel_boundary2(:,2)).^2));
                    [~,tri_boundary2]=min(sqrt((tri_mouth(2,1)-mainchannel_boundary2(:,1)).^2+...
                        (tri_mouth(2,2)-mainchannel_boundary2(:,2)).^2));
                    tri_boundary_idx=[tri_boundary1,tri_boundary2];

                    tri_boundary_t=mainchannel_boundary2(min(tri_boundary_idx):max(tri_boundary_idx),:);
                    in=inpolygon(vp(temp_path,1),vp(temp_path,2),tri_boundary_t(:,1),tri_boundary_t(:,2));
                    if sum(in)/length(temp_path)<=0.8 % if less than 80% centerline in the tributary polygon
                        tri_boundary_t=setdiff(mainchannel_boundary2,tri_boundary_t,'rows','stable');
                    end

                    if isequal([tri_boundary_t(end,1),tri_boundary_t(end,2)],[tri_boundary_t(1,1),tri_boundary_t(1,2)])
                    else
                        tri_boundary_t(end+1,:)=[tri_boundary_t(1,1),tri_boundary_t(1,2)];
                    end
                    data.data_bdy.order2.(['t',num2str(po,['%0',num2str(num_digit+1),'.f'])])=tri_boundary_t;% save the tributary boundary to tributary.tb**
                    mainchannel_boundary2=setdiff(mainchannel_boundary2,tri_boundary_t,'rows','stable');
                    data.data_ctl.order2.(['t',num2str(po,['%0',num2str(num_digit+1),'.f'])])=vp(temp_path,:);% save the tributary centerline to tributary.tc**
                    break;
                end
            else
                min_r=min_r+1;
            end
        end

    end

end
fprintf('The order2 centerline and bounday extracted, %.2f s!\n',toc)

%% use the cirle of connection3 points to search the nearest order 3 channel mouth, to get the order channel3 boundary (order 2)
% order3 centerline and boundary
disp('Extracting the order3 centerline and bounday...')
tic
if ~isempty(path_left_3connections)
    for po=1:size(path_left_3connections,2)

        temp_path=path_left_3connections(:,po);
        temp_path(temp_path==0,:)=[];
        order3_connection3=left_points(sum_temp_left==3);
        if isempty(order3_connection3)
            break;% only left one path to search
        end
        num_digit=numel(num2str(size(path_left_3connections,2)));
        [idx0,idx]=ismember(order3_connection3,temp_path);
        idx0=find(idx0);
        if length(idx0)==length(order3_connection3) % if the left points only two and all belong to order3_connection3
            mainchannel_boundary2(end+1,:)=mainchannel_boundary2(1,:);
            data.data_bdy.order3.(['t',num2str(po,['%0',num2str(num_digit+1),'.f'])])=mainchannel_boundary2;% save the tributary boundary to tributary.tb**
            data.data_ctl.order3.(['t',num2str(po,['%0',num2str(num_digit+1),'.f'])])=vp(temp_path,:);% save the tributary centerline to tributary.tc**
            break;
        end
        center=vp(order3_connection3(idx0),:);
        min_r=min(sqrt((center(1,1)-mainchannel_boundary2(:,1)).^2+(center(1,2)-mainchannel_boundary2(:,2)).^2));

        th = linspace(0,2*pi) ;

        for search=1:20 % search for no more than 20 times
            cc_x = center(1)+min_r*cos(th) ;
            cc_y = center(2)+min_r*sin(th) ;
            % intersection of circle with boundary
            [c_b_x,c_b_y] = intersections(cc_x,cc_y,mainchannel_boundary2(:,1),mainchannel_boundary2(:,2),1);
            % intersection of circle with tributary centerline(tc)
            [c_tc_x,c_tc_y] = intersections(vp(temp_path,1),vp(temp_path,2),cc_x,cc_y,1);
            if isempty(c_tc_x)||isempty(c_b_x)
                min_r=min_r+1;
                continue;
            end
            k_c_tc=(center(2)-c_tc_y)/(center(1)-c_tc_x);
            b_c_tc=center(2)-k_c_tc*center(1);
            d_temp=k_c_tc*c_b_x+b_c_tc-c_b_y;% find the location of bdy intersection to the line of center to intersection of circle with tributary centerline(tc)
            d_temp1=find(d_temp<=0);
            d_temp2=find(d_temp>0);

            % get the angle of intersection of circle/tri_centerline to cirlce center
            th_c_tc=atan2d((c_tc_y-center(2)),(c_tc_x-center(1)));
            if th_c_tc<0
                th_c_tc=th_c_tc+360;
            end
            % get the angle of intersec of circle/boundary to circle center
            th_c_b=atan2d((c_b_y-center(2)),(c_b_x-center(1)));
            if th_c_tc>=0&&th_c_tc<90

            elseif th_c_tc>=90&&th_c_tc<270
                th_c_b(th_c_b<0)=th_c_b(th_c_b<0)+360;
            elseif th_c_tc>=270&&th_c_tc<360
                th_c_tc=th_c_tc-360;
            end
            if length(th_c_b)>=2 && ~isempty(th_c_b(th_c_b>=th_c_tc))&& ~isempty(th_c_b(th_c_b<th_c_tc))&&~isempty(d_temp1)&&~isempty(d_temp2)
                diff_th=th_c_tc-th_c_b;
                diff_th=[d_temp,diff_th];
                diff_th1=diff_th(d_temp1,:);
                diff_th2=diff_th(d_temp2,:);
                [~,min_th1]=min(abs(diff_th1(:,2)));
                [~,min_th2]=min(abs(diff_th2(:,2)));
                [~,min_th1_idx]=ismember(diff_th1(min_th1,:),diff_th,'rows');
                [~,min_th2_idx]=ismember(diff_th2(min_th2,:),diff_th,'rows');
                tri_mouth=[c_b_x([min_th1_idx;min_th2_idx]),c_b_y([min_th1_idx;min_th2_idx])];
                [~,tri_boundary1]=min(sqrt((tri_mouth(1,1)-mainchannel_boundary2(:,1)).^2+...
                    (tri_mouth(1,2)-mainchannel_boundary2(:,2)).^2));
                [~,tri_boundary2]=min(sqrt((tri_mouth(2,1)-mainchannel_boundary2(:,1)).^2+...
                    (tri_mouth(2,2)-mainchannel_boundary2(:,2)).^2));
                tri_boundary_idx=[tri_boundary1,tri_boundary2];

                tri_boundary_t=mainchannel_boundary2(min(tri_boundary_idx):max(tri_boundary_idx),:);
                in=inpolygon(vp(temp_path,1),vp(temp_path,2),tri_boundary_t(:,1),tri_boundary_t(:,2));
                if sum(in)/length(temp_path)<=0.8 % if less than 80% centerline in the tributary polygon
                    tri_boundary_t=setdiff(mainchannel_boundary2,tri_boundary_t,'rows','stable');
                end

                if isequal([tri_boundary_t(end,1),tri_boundary_t(end,2)],[tri_boundary_t(1,1),tri_boundary_t(1,2)])
                else
                    tri_boundary_t(end+1,:)=[tri_boundary_t(1,1),tri_boundary_t(1,2)];
                end
                data.data_bdy.order3.(['t',num2str(po,['%0',num2str(num_digit+1),'.f'])])=tri_boundary_t;% save the tributary boundary to tributary.tb**
                mainchannel_boundary2=setdiff(mainchannel_boundary2,tri_boundary_t,'rows','stable');
                data.data_ctl.order3.(['t',num2str(po,['%0',num2str(num_digit+1),'.f'])])=vp(temp_path,:);% save the tributary centerline to tributary.tc**

                break;
            else
                min_r=min_r+1;
            end
        end

    end
end
fprintf('The order3 centerline and bounday extracted, %.2f s!\n',toc)



%%  get the transects of each segments channel

disp('Calculating the transects width...')
tic
fldnm1=fieldnames(data.data_ctl);
transec_width_total=[];

for jjj=1:length(fldnm1)
    fldnm2=fieldnames(getfield(data.data_ctl,fldnm1{jjj}));
    for kkk=1:length(fldnm2)
        cl_fit=eval(['data.data_ctl.',fldnm1{jjj},'.',fldnm2{kkk},';']);
        boundary1=eval(['data.data_bdy.',fldnm1{jjj},'.',fldnm2{kkk},';']);

        plot(boundary1(:,1),boundary1(:,2))
        hold on
        plot(cl_fit(:,1),cl_fit(:,2),'-g','linewidth',1)

        transec_width=cal_width(cl_fit,boundary1,spacing);% transects spacing 10 units

        data_export=transec_width;
        transec_width_total=[transec_width_total;transec_width];
        data_export(:,6)=(data_export(:,1)+data_export(:,3))/2;
        data_export(:,7)=(data_export(:,2)+data_export(:,4))/2;
        data_export=array2table(data_export,'VariableNames',{'x1','y1','x2','y2','width','x_c','y_c'});
        eval(['data.data_width.',fldnm1{jjj},'.',fldnm2{kkk},'=data_export;']);
        width75th=prctile(data_export.width,75);
        eval(['data.data_width75th.',fldnm1{jjj},'.',fldnm2{kkk},'=width75th;']);

    end
end

fprintf('Segemnt channel width calculated and exported, %.2f s\n',toc)


%% process island

island_num=length(island);

if ~isempty(island_num)&&island_num~=0
    islandWidth=zeros(size(transec_width_total,1),island_num);
    temp_island=[];
    tic
    num_transec_width_total=size(transec_width_total,1);
    plot(mainchannel_boundary(:,1),mainchannel_boundary(:,2)  )
    hold on
    for p=1:island_num
        tempisland=B_island{island(p)};
        plot(tempisland(:,1),tempisland(:,2),'-m',...
            'LineWidth',1.5)
    end
    parfor t=1:num_transec_width_total
        for p=1:island_num
            disp([num2str(t),'/',num2str(p)])
            kcs=(transec_width_total(t,2)-transec_width_total(t,4))/(transec_width_total(t,1)-transec_width_total(t,3));
            bcs=transec_width_total(t,2)-kcs*transec_width_total(t,1);
            
            tempisland=B_island{island(p)};
                    plot(tempisland(:,1),tempisland(:,2),'-m',...
                        'LineWidth',1.5)

            [x,y]=find_intersection_kiki_sort([tempisland;tempisland(1,:)],kcs,bcs);% make the polyline of tempisland close
            if ~isempty(x) && max(x)<=max(transec_width_total(t,1),transec_width_total(t,3))&& min(x)>=min(transec_width_total(t,1),transec_width_total(t,3))...
                    &&max(y)<=max(transec_width_total(t,2),transec_width_total(t,4))&&min(y)>=min(transec_width_total(t,2),transec_width_total(t,4))

                if length(x)==1
                elseif length(x)==2
                    islandWidth(t,p)=abs(norm([x(1),y(1)]-[x(2),y(2)]));% calculate the island width
                elseif length(x)==3
        

                    center12=[(x(1)+x(2))/2,(y(1)+y(2))/2];
                    center23=[(x(3)+x(2))/2,(y(3)+y(2))/2];

                    islandPolygon=[tempisland;tempisland(1,:)];
                    islandPolygon12=inpolygon(center12(1),center12(2),islandPolygon(:,1),islandPolygon(:,2));
                    islandPolygon23=inpolygon(center23(1),center23(2),islandPolygon(:,1),islandPolygon(:,2));
                    islandWidth(t,p)=abs(norm([x(1),y(1)]-[x(2),y(2)]))*islandPolygon12+...
                        abs(norm([x(2),y(2)]-[x(3),y(3)]))*islandPolygon23;

                    disp(['Need to pay attention that there are THREE intersections!',' Transec number is: ',num2str(t),', Island number is ',num2str(p)]);
                    temp_island=[temp_island;[t,p,3]];
                elseif length(x)==4
  
                    center12=[(x(1)+x(2))/2,(y(1)+y(2))/2];
                    center23=[(x(3)+x(2))/2,(y(3)+y(2))/2];
                    center34=[(x(3)+x(4))/2,(y(3)+y(4))/2];
                    islandPolygon=[tempisland;tempisland(1,:)];
                    islandPolygon12=inpolygon(center12(1),center12(2),islandPolygon(:,1),islandPolygon(:,2));
                    islandPolygon23=inpolygon(center23(1),center23(2),islandPolygon(:,1),islandPolygon(:,2));
                    islandPolygon34=inpolygon(center34(1),center34(2),islandPolygon(:,1),islandPolygon(:,2));
                    islandWidth(t,p)=abs(norm([x(1),y(1)]-[x(2),y(2)]))*islandPolygon12+...
                        abs(norm([x(2),y(2)]-[x(3),y(3)]))*islandPolygon23+...
                        abs(norm([x(3),y(3)]-[x(4),y(4)]))*islandPolygon34;

                    disp(['Need to pay attention that there are FOUR intersections!',' Transec=',num2str(t),', Island number is',num2str(p)]);

                    temp_island=[temp_island;[t,p,4]];
                else
                    islandWidth(t,p)=max(pdist([x,y],'euclidean'));
                    disp(['Need to pay attention that there are More Than Four intersections!',' Transec=',num2str(t),', Island number is',num2str(p)]);
                end
            else
            end

        end

    end
else
    islandWidth=0;
end
transection_width_final=transec_width_total(:,5)-sum(islandWidth,2);
transection_width_final2=[(transec_width_total(:,1)+transec_width_total(:,3))/2,(transec_width_total(:,2)+transec_width_total(:,4))/2,transection_width_final];
centerline_x=transection_width_final2(:,1);
centerline_y=transection_width_final2(:,2);
cs_width=transection_width_final2(:,3);
width_withIsland=[transec_width_total,transection_width_final,centerline_x,centerline_y];

writetbl=array2table(width_withIsland,'VariableNames',{'x1','y1','x2','y2','width_noisland','width_island','x_c','y_c'});
fprintf('Calculated the width include the influence of islands by %.2f seconds!!!\n',toc)
te=toc(ts);
disp(['Total time: ',num2str(te)]);
end



%% sub functions

function next_point_forward_output=findNextPoint_forward(current_search,K_new,scan_mark,ii,cc_1,river_path2)

[loc_current_row,loc_current_col]=find(K_new==current_search);
next_point_all=zeros(length(loc_current_row),1);
for t=1:length(loc_current_row)
    if loc_current_col(t)==1
        next_point_col=2;
    else
        next_point_col=1;
    end
    next_point_all(t)=K_new(loc_current_row(t),next_point_col);
end
if min(scan_mark(next_point_all))==0
    if sum(scan_mark(next_point_all)==0)==2 % if the connection two nodes all did not forward scan before, use previous path to identify
        next_point=next_point_all(next_point_all~=river_path2(ii-1,cc_1));
    else
        next_point=next_point_all(scan_mark(next_point_all)==0);
    end
else
    next_point=next_point_all(scan_mark(next_point_all)>scan_mark(current_search));
end

if length(next_point)==2 && (scan_mark(next_point(1))~= scan_mark(next_point(2)))
    [~,temp]=max(scan_mark(next_point));
    next_point=next_point(temp);
else
    if ~isempty(next_point)
    next_point=next_point(1);% random choose one as next point
    end
end
if length(next_point_all)==1
    next_point=next_point_all;
end
    
next_point_forward_output=next_point;
end


function next_point_backward_output=findNextPoint_backward(current_search,K,forward_mark)

[loc_current_row,loc_current_col]=find(K==current_search);
next_point_all=zeros(length(loc_current_row),1);
for t=1:length(loc_current_row)
    if loc_current_col(t)==1
        next_point_col=2;
    else
        next_point_col=1;
    end
    next_point_all(t)=K(loc_current_row(t),next_point_col);
end
if min(forward_mark(next_point_all))==1
    if sum(forward_mark(next_point_all)==1)==2 % if the connection two nodes all delete at the first scan
        random_choose=next_point_all(forward_mark(next_point_all)==1);
        next_point=random_choose(1); % random choose one next point
    else
        next_point=next_point_all(forward_mark(next_point_all)~=1 & forward_mark(next_point_all)<forward_mark(current_search));
        if isempty(next_point)
            next_point=next_point_all(forward_mark(next_point_all)<forward_mark(current_search));
        end
    end
else
    next_point=next_point_all(forward_mark(next_point_all)<forward_mark(current_search));
end


if length(next_point)==2 && (forward_mark(next_point(1))~= forward_mark(next_point(2)))
    [~,temp]=max(forward_mark(next_point));
    next_point=next_point(temp);
else
    if ~isempty(next_point)
        next_point=next_point(1);% random choose one as next point
    end
end
next_point_backward_output=next_point;
end


function [NextPoint,K]=FindNextPoint(current_point,K)
currentpoint_index=find(K(:,1)==current_point);
if isempty(currentpoint_index)
    currentpoint_index=find(K(:,2)==current_point);
    NextPoint=K(currentpoint_index,1);
    K(currentpoint_index,1)=0;
else
    NextPoint=K(currentpoint_index,2);
    K(currentpoint_index,2)=0;
end

end

function [x,y]=find_intersection_kiki(boundary,kk,bb)
% find the intersection of line with scatter
pos_b=kk*boundary(:,1)-boundary(:,2)+bb;
num_b=length(pos_b);
result=nan(num_b,2);
for ii=1:num_b
    if pos_b(ii)==0
        result(ii,:)=boundary(ii,:);
    elseif ii<num_b && pos_b(ii)*pos_b(ii+1)<0
        x1=boundary(ii,1);
        y1=boundary(ii,2);
        x2=boundary(ii+1,1);
        y2=boundary(ii+1,2);
        if x1==x2
            k=99999;
        else
            k=(y1-y2)/(x1-x2);
        end
        b=y1-k*x1;
        x_0=(b-bb)/(kk-k);
        y_0=k*x_0+b;
        result(ii,:)=[x_0,y_0];
    end
end
result(isnan(result(:,1)),:)=[];
% %  this part is used to sort the result-------------
% if ~isempty(result)
%     temp_dist=squareform(pdist(result(:,1:2),'euclidean'));
%     [I,~]=find(temp_dist==max(temp_dist));
%     p0=result(I(1),:);
%     result_temp=abs(norm([result(:,1),result(:,2)]-[p0(1),p0(2)]));
%     result=result(result,result_temp);
%     result=sortrows(result,3);
% end
% %--------------------------
x=result(:,1);
y=result(:,2);




end

function [x,y]=find_intersection_kiki_sort(boundary,kk,bb)
% find the intersection of line with scatter
pos_b=kk*boundary(:,1)-boundary(:,2)+bb;
num_b=length(pos_b);
result=nan(num_b,2);
for ii=1:num_b
    if pos_b(ii)==0
        result(ii,:)=boundary(ii,:);
    elseif ii<num_b && pos_b(ii)*pos_b(ii+1)<0
        x1=boundary(ii,1);
        y1=boundary(ii,2);
        x2=boundary(ii+1,1);
        y2=boundary(ii+1,2);
        if x1==x2
            k=99999;
        else
            k=(y1-y2)/(x1-x2);
        end
        b=y1-k*x1;
        x_0=(b-bb)/(kk-k);
        y_0=k*x_0+b;
        result(ii,:)=[x_0,y_0];
    end
end
result(isnan(result(:,1)),:)=[];
%  this part is used to sort the result-------------
if ~isempty(result)
    temp_dist=squareform(pdist(result(:,1:2),'euclidean'));
    [I,~]=find(temp_dist==max(temp_dist));
    p0=result(I(1),:);
    result_temp=sqrt((result(:,1)-p0(1)).^2+(result(:,2)-p0(2)).^2);
    result=[result,result_temp];
    result=sortrows(result,3);
end
%--------------------------
x=result(:,1);
y=result(:,2);
end

function transec_width=cal_width(cl_fit,boundary1,spacing)
% given the channel boundary and centerline to calculate the width
% cl_fit, the centerline coordinates,[x,y]
% boundary1, the closed boundary coordinates, [x,y]
% spacing, the spacing of transects
smoothedData = smoothdata(cl_fit,"gaussian",10);% use Gaussian, moving window 10 to filter
dis_total=sum(hypot(diff(cl_fit(:,1)),diff(cl_fit(:,2))));% calculate the centerline path distance

cl_fit=smoothedData;% fit line
if dis_total>=spacing
else
    spacing=1;% if the centerline distance shorter than spacing, set spacing as 1
end
pt = interparc(ceil(dis_total/spacing),cl_fit(:,1),cl_fit(:,2),'spline');% get the same spacing point

cl_fit=pt;
transec_width=zeros(size(cl_fit,1),5);
tic
for ii=1:size(cl_fit,1)
    %     disp(num2str(ii))
    cl=cl_fit;
    cl_x=cl_fit(ii,1);
    cl_y=cl_fit(ii,2);
    %     p=polyfit(cl(ii-1:ii+1,1),cl(ii-1:ii+1,2),1);
    if ii==1
        k=(cl(ii+1,2)-cl(ii,2))/(cl(ii+1,1)-cl(ii,1));
    elseif ii==size(cl_fit,1)
        k=(cl(ii,2)-cl(ii-1,2))/(cl(ii,1)-cl(ii-1,1));
    else
        k=(cl(ii+1,2)-cl(ii-1,2))/(cl(ii+1,1)-cl(ii-1,1));
    end

    b=cl_y-k*cl_x;
    if k==0
        kk=99999;%kk is the gradient of the line perpendicular to the centerline
    else
        kk=-1/k;
    end
    bb=cl_y-kk*cl_x;
    [x_i,y_i]=find_intersection_kiki(boundary1,kk,bb);
    d_temp=k*x_i+b-y_i;
    x1=x_i(d_temp>=0);
    y1=y_i(d_temp>=0);
    x2=x_i(d_temp<0);
    y2=y_i(d_temp<0);

    if ~isempty (x1) && ~isempty (x2) % find the points of both side closest to center point
        [dist1,dist1_idx]=min(sqrt((x1-cl_x).^2+(y1-cl_y).^2));
        [dist2,dist2_idx]=min(sqrt((x2-cl_x).^2+(y2-cl_y).^2));
        cs_pair_temp=[x1(dist1_idx),y1(dist1_idx),x2(dist2_idx),y2(dist2_idx)];
        transec_width(ii,:)=[cs_pair_temp,dist1+dist2];
    end
%     line([cs_pair_temp(1),cs_pair_temp(3)],[cs_pair_temp(2),cs_pair_temp(4)])
%     drawnow

end
%
transec_width(transec_width(:,5)==0,:)=[];
toc
end

function path_sort=path_sort_fun(river_path_tributary_test,start_point,vp)
% river_path_tributary_test is index to vp
% start_point is the first point, coordinates,[x y]
% input river_path_tributary_test is index, then output path_sort is index
% input river_path_tributary_test is coordinates, then output path_sort is coordinates
% river_path_tributary_test(river_path_tributary_test==0)=[];
% river_path_tributary_test=sort(river_path_tributary_test);
% numPoints=size(river_path_tributary_test,1);
if size(river_path_tributary_test,2)==2
    river_path_tributary_test(river_path_tributary_test(:,1)==0,:)=[];
    river_path_tributary_test=sortrows(river_path_tributary_test,1);
    numPoints=size(river_path_tributary_test,1);

    x=river_path_tributary_test(:,1);
    y=river_path_tributary_test(:,2);
    [~,currentIndex]=ismember(start_point,river_path_tributary_test,'rows'); % find the stat point index of current dataset
else
    river_path_tributary_test(river_path_tributary_test==0)=[];
    river_path_tributary_test=sort(river_path_tributary_test);
    numPoints=size(river_path_tributary_test,1);
    x=vp(river_path_tributary_test,1);
    y=vp(river_path_tributary_test,2);
    [~,currentIndex]=ismember(start_point,vp(river_path_tributary_test,:),'rows'); % find the stat point index of current dataset
end

% Make a list of which points have been visited
beenVisited = false(1, numPoints);
% Make an array to store the order in which we visit the points.
visitationOrder = ones(1, numPoints);
% Define a filasafe
maxIterations = numPoints + 1;
iterationCount = 1;
% Visit each point, finding which unvisited point is closest.
% Define a current index.  currentIndex will be 1 to start and then will vary.
% currentIndex = 1;

% [~,currentIndex]=ismember(start_point,vp(river_path_tributary_test,:),'rows'); % find the stat point index of current dataset
% currentIndex = 482; % modify here



while sum(beenVisited) < numPoints && iterationCount < maxIterations
    % Indicate current point has been visited.
    visitationOrder(iterationCount) = currentIndex;
    beenVisited(currentIndex) = true;
    % Get the x and y of the current point.
    thisX = x(currentIndex);
    thisY = y(currentIndex);
    %   text(thisX + 0.01, thisY, num2str(currentIndex), 'FontSize', 35, 'Color', 'r');
    % Compute distances to all other points
    distances = sqrt((thisX - x) .^ 2 + (thisY - y) .^ 2);
    % Don't consider visited points by setting their distance to infinity.
    distances(beenVisited) = inf;
    % Also don't want to consider the distance of a point to itself, which is 0 and would alsoways be the minimum distances of course.
    distances(currentIndex) = inf;
    % Find the closest point.  this will be our next point.
    %         [minDistance, indexOfClosest] = min(distances);
    [~, indexOfClosest] = min(distances);
    % Save this index
    iterationCount = iterationCount + 1;
    % Set the current index equal to the index of the closest point.
    currentIndex = indexOfClosest;
end
path_sort=river_path_tributary_test(visitationOrder,:);
%     scatter(vp(river_path_tributary_test,1),vp(river_path_tributary_test,2),20,[0.5 0.5 0.5],'filled')
%     hold on
%     plot(vp(path_sort,1),vp(path_sort,2),'-g','LineWidth',2);
end

