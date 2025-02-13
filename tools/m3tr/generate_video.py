import os.path as osp
import argparse
import os
import glob
import cv2
import mmcv

def parse_args():
    parser = argparse.ArgumentParser(description='vis hdmaptr map gt label')
    parser.add_argument('visdir', help='visualize directory')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--fps', default=10, type=int, help='fps to generate video')
    parser.add_argument('--video-name', default='demo',type=str)
    parser.add_argument('--gt-map-name', default='GT_MAP.png', type=str)
    parser.add_argument('--pred-map-name', default='PRED_MAP.png', type=str)
    parser.add_argument('--surr-view-name', default='surround_view.jpg', type=str)
    parser.add_argument('--sample-name', default='SAMPLE_VIS.jpg', type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    parent_dir = osp.join(args.visdir,'..')
    vis_subdir_list = []
    # import pdb;pdb.set_trace()
    size = (2560,686)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_path = osp.join(parent_dir,'%s.mp4' % args.video_name)
    video = cv2.VideoWriter(video_path, fourcc, args.fps, size, True)
    file_list = os.listdir(args.visdir)
    file_list.sort()
    prog_bar = mmcv.ProgressBar(len(file_list))
    for file in file_list:
        file_path = osp.join(args.visdir, file) 
        if os.path.isdir(file_path):
            vis_subdir_list.append(file_path)

            gt_path = osp.join(file_path,args.gt_map_name)
            pred_path = osp.join(file_path,args.pred_map_name)
            surr_path = osp.join(file_path,args.surr_view_name) 
            sample_path = osp.join(file_path,args.sample_name)
            
            map_img = cv2.imread(pred_path)
            gt_map_img = cv2.imread(gt_path)
            cams_img = cv2.imread(surr_path)

            map_img = cv2.copyMakeBorder(map_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)
            gt_map_img = cv2.copyMakeBorder(gt_map_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)
            map_img = cv2.rotate(map_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            gt_map_img = cv2.rotate(gt_map_img, cv2.ROTATE_90_COUNTERCLOCKWISE)


            cams_h,cam_w,_ = cams_img.shape
            map_h,map_w,_ = map_img.shape
            resize_ratio = cams_h / map_h
            resized_w = map_w * resize_ratio
            resized_map_img = cv2.resize(map_img,(int(resized_w),int(cams_h)))
            resized_gt_map_img = cv2.resize(gt_map_img,(int(resized_w),int(cams_h)))

            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # fontScale
            fontScale = 4
            # Line thickness of 2 px
            thickness = 10
            # org
            org = (20, 100)      
            # Blue color in BGR
            color = (0, 0, 255)
            # Using cv2.putText() method
            resized_map_img = cv2.putText(resized_map_img, 'PRED', org, font, 
                            fontScale, color, thickness, cv2.LINE_AA)
            resized_gt_map_img = cv2.putText(resized_gt_map_img, 'GT', org, font, 
                            fontScale, color, thickness, cv2.LINE_AA)
            
            sample_img = cv2.hconcat([cams_img, resized_map_img, resized_gt_map_img])
            cv2.imwrite(sample_path, sample_img,[cv2.IMWRITE_JPEG_QUALITY, 70])
            # import pdb;pdb.set_trace()
            resized_img = cv2.resize(sample_img,size)

            video.write(resized_img)
        prog_bar.update()
    # import pdb;pdb.set_trace()
    video.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

