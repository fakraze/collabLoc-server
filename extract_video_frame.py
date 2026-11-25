import cv2
import argparse

# Extract video frames
def main():
    parser = argparse.ArgumentParser(description="從影片中擷取影格並儲存為圖片的工具")
    
    # set arguments
    parser.add_argument('--gap', type=int, default=20, help='每隔多少幀擷取一張 (預設: 20)')
    parser.add_argument('--input_dir', type=str, required=True, help='影片來源目錄路徑')
    parser.add_argument('--output_dir', type=str, required=True, help='圖片輸出目錄路徑')
    parser.add_argument('--video_pre', type=str, required=True, help='影片檔名或檔案前綴')
    parser.add_argument('--start_idx', type=int, default=0, help='影片起始索引(預設:0)')
    parser.add_argument('--end_idx', type=int, default=1, help='影片結束索引(預設:1)')
    parser.add_argument('--output_pre', type=str, default='db_', help='輸出圖片前綴 (預設: db_)')
    parser.add_argument('--show', action='store_true', help='是否顯示預覽視窗 (預設: 不顯示)')

    args = parser.parse_args()
    
    false_cnt = 0
    for video_idx in range(args.start_idx, args.end_idx):
        frame_idx = 0
        print(f'Processing {args.video_pre}{video_idx}.MP4')

        cap = cv2.VideoCapture(f'{args.input_dir}/{args.video_pre}{video_idx}.mp4')
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_idx += 1

                if frame_idx % args.gap == 1:
                    if args.show:
                        cv2.imshow('frame', frame)
                        cv2.waitKey(1)
                    frame_idx_str = str(frame_idx).zfill(5)
                    cv2.imwrite(f'{args.output_dir}/{args.output_pre}{args.video_pre}{video_idx}_{frame_idx_str}.jpg', frame)

                false_cnt = 0
            else:
                false_cnt += 1
                if false_cnt > 300:
                    cap.release()
                    break


if __name__ == '__main__':
    main()