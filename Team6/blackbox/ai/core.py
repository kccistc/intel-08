import queue
import numpy as np
import async_api
import os
import time, uuid
import pre_post_process
import queue as pyqueue




def backbone_raw_data(target, input_path, hef_path, queue_out, queue_meta_out, demo, scenes, run_slow) -> None:
    """
    Runs inference on the backbone model asynchronously, processing a series of images from the nuScenes dataset.

    This function handles the loading and preprocessing of images from the nuScenes dataset,
    performs asynchronous inference on the specified backbone model using the HailoAsyncInference API,
    and manages the flow of metadata and results through the provided queues. The function runs continuously
    in a loop, processing scenes and samples, and controlling the inference speed based on the `run_slow` flag.

    Args:
        target (str): The target device for inference (e.g., a Hailo device).
        data_path (str): The path to the dataset containing the image files.
        hef_path (str): The path to the Hailo Execution Flow (HEF) file.
        queue_out (Queue): The output queue for inference results.
        queue_meta_out (Queue): The output queue for metadata.
        demo (object): A demo object that handles termination signals and visualization.
        run_slow (bool): If `True`, the inference runs at 5 FPS, slowing the processing speed.
        nusc (object): The nuScenes dataset object that provides access to samples and scene data.

    Returns:
        None: This function runs indefinitely until the termination signal is received from the `demo` object.
    """
    assert os.path.exists(hef_path), f"File not found: {hef_path}"

    hailo_inference = async_api.HailoAsyncInference(target, hef_path, queue_out, demo,
                                          ['petrv2_repvggB0_backbone_pp_800x320/input_layer1'],
                                          ['petrv2_repvggB0_backbone_pp_800x320/conv28'], 6,  output_type='FLOAT32', input_type='UINT8')
    tensor_data = []
    tokens = []
    for scene in scenes:
        assert 'input' in scene, "Please run ./prepare_data.py script with --raw-data flag"
        input_file_name = scene['input']
        tensor_data.append(np.load(f'{input_path}/{input_file_name}'))
        tokens.append(scene['tokens'])

    while True:
        last_timestamp = time.time()
        for scene_tokens, tensor_datas in zip(tokens, tensor_data):
            for i, token in enumerate(scene_tokens):
                if run_slow:
                    while(time.time() - last_timestamp < (1/5)):
                        time.sleep(0.002)
                    last_timestamp = time.time()

                job = hailo_inference.run({'petrv2_repvggB0_backbone_pp_800x320/input_layer1': tensor_datas[i]})
                while not demo.get_terminate():
                    try:
                        queue_meta_out.put(token, block=True, timeout=0.5)
                        break
                    except queue.Full:
                        pass

                if demo.get_terminate():
                    job.wait(100000)
                    return
                
import numpy as np
from queue import Queue as PyQueue
import queue as pyqueue
import time, os

def backbone_from_cam(target, frames_src_q, hef_path, queue_out, queue_meta_out, demo, run_slow):
    assert os.path.exists(hef_path), f"File not found: {hef_path}"
    import async_api
    print(f"[Backbone] Process started...")

    # HailoAsyncInference를 초기화할 때 batch_size를 6으로 설정합니다.
    # 이렇게 하면 6개 이미지를 한 번의 추론 작업으로 처리할 수 있습니다.
    hailo_inference = async_api.HailoAsyncInference(
        target, hef_path, queue_out, demo,  # ★ 출력을 다음 스테이지 큐(queue_out)로 직접 보냅니다.
        ['petrv2_repvggB0_backbone_pp_800x320/input_layer1'],
        ['petrv2_repvggB0_backbone_pp_800x320/conv28'],
        6,  # ★★★ 중요: batch_size를 6으로 변경
        output_type='FLOAT32', input_type='UINT8'
    )

    last_ts = time.time()
    while not demo.get_terminate():
        try:
            # (6, 320, 800, 3) 형태의 numpy 배열과 메타데이터를 가져옵니다.
            frames_np, meta = frames_src_q.get(timeout=0.5)
        except pyqueue.Empty:
            continue
        
        # 6개의 프레임이 모두 있는지 확인
        if frames_np.shape[0] != 6:
            print(f"[Backbone] WARN: Expected 6 frames, but got {frames_np.shape[0]}. Skipping.")
            continue

        # 옵션: 속도 제한
        if run_slow:
            now = time.time()
            dt = now - last_ts
            if dt < 1/5:
                time.sleep((1/5) - dt)
            last_ts = time.time()

        # === 6개 이미지를 하나의 배치로 묶어 한 번에 추론 ===
        # 더 이상 반복문으로 하나씩 처리하고 결과를 기다릴 필요가 없습니다.
        _ = hailo_inference.run({
            'petrv2_repvggB0_backbone_pp_800x320/input_layer1': frames_np
        })

        # 메타데이터도 다음 스테이지로 즉시 전달합니다.
        # 추론 결과(payload)는 HailoAsyncInference의 콜백 함수가
        # 알아서 queue_out으로 넣어주므로 여기서 신경 쓸 필요가 없습니다.
        pushed = False
        while not demo.get_terminate() and not pushed:
            try:
                queue_meta_out.put(meta, timeout=0.5)
                pushed = True
            except pyqueue.Full:
                pass
    
    print("[Backbone] Process finished.")


        
def backbone_from_jpg(target, data_path, hef_path, queue_out, queue_meta_out, demo, scenes, run_slow, nusc) -> None:
    """
    Runs inference on the backbone model asynchronously, processing a series of images from the nuScenes dataset.

    This function handles the loading and preprocessing of images from the nuScenes dataset,
    performs asynchronous inference on the specified backbone model using the HailoAsyncInference API,
    and manages the flow of metadata and results through the provided queues. The function runs continuously
    in a loop, processing scenes and samples, and controlling the inference speed based on the `run_slow` flag.

    Args:
        target (str): The target device for inference (e.g., a Hailo device).
        data_path (str): The path to the dataset containing the image files.
        hef_path (str): The path to the Hailo Execution Flow (HEF) file.
        queue_out (Queue): The output queue for inference results.
        queue_meta_out (Queue): The output queue for metadata.
        demo (object): A demo object that handles termination signals and visualization.
        run_slow (bool): If `True`, the inference runs at 5 FPS, slowing the processing speed.
        nusc (object): The nuScenes dataset object that provides access to samples and scene data.

    Returns:
        None: This function runs indefinitely until the termination signal is received from the `demo` object.
    """
    assert os.path.exists(hef_path), f"File not found: {hef_path}"

    hailo_inference = async_api.HailoAsyncInference(target, hef_path, queue_out, demo,
                                          ['petrv2_repvggB0_backbone_pp_800x320/input_layer1'],
                                          ['petrv2_repvggB0_backbone_pp_800x320/conv28'], 6,  output_type='FLOAT32', input_type='UINT8')
    while True:
        last_timestamp = time.time()
        for scene in scenes:
            scene_tokens = scene['tokens']
            for token in scene_tokens:

                file_paths = []
                for cam in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
                    file_paths.append(nusc[token][cam][2]['filename'])
                if run_slow:
                    while(time.time() - last_timestamp < (1/5)):
                        time.sleep(0.002)
                    last_timestamp = time.time()

                job = hailo_inference.run({'petrv2_repvggB0_backbone_pp_800x320/input_layer1': pre_post_process.preprocess(data_path, file_paths)})
                while not demo.get_terminate():
                    try:
                        queue_meta_out.put(token, block=True, timeout=0.5)
                        break
                    except queue.Full:
                        pass

                if demo.get_terminate():
                    job.wait(100000)
                    return
from queue import Queue as PyQueue
import queue as pyqueue
import time, os, numpy as np

def transformer(target, hef_path, matmul_path, queue_in,
                queue_meta_in, queue_out, queue_meta_out, demo, alpha = 1.0, beta = 0.0) -> None:
    assert os.path.exists(hef_path)
    assert os.path.exists(matmul_path)
    
    print(f"[Transformer] input queue size : {queue_in.qsize()} ...")
    
    # ★ 콜백은 여기로만 넣게 한다
    local_out_q = PyQueue(maxsize=8)

    hailo_inference = async_api.HailoAsyncInference(
        target, hef_path, local_out_q, demo,             # ← queue_out 말고 local_out_q
        ['petrv2_repvggB0_transformer_pp_800x320/input_layer1',
         'petrv2_repvggB0_transformer_pp_800x320/input_layer2'],
        ['petrv2_repvggB0_transformer_pp_800x320/concat1',
         'petrv2_repvggB0_transformer_pp_800x320/conv41'],
        1
    )

    matmul = np.load(matmul_path)
    assert matmul.shape == (1, 12, 250, 256)
    matmul = (alpha * matmul + beta).astype(np.float32, copy=False)
    prev_block = None
    while not demo.get_terminate():
        # 메타 / 인풋 받기
        try:
            meta_data = queue_meta_in.get(timeout=0.5)
        except pyqueue.Empty:
            continue
        try:
            in_data = queue_in.get(timeout=0.5)
        except pyqueue.Empty:
            continue

        mid1_list = in_data['petrv2_repvggB0_backbone_pp_800x320/conv28']
        assert len(mid1_list) == 6 and mid1_list[0].shape == (10, 25, 1280)

        cur_block = np.stack(mid1_list, axis=0)
        mid2_block = np.concatenate([cur_block, (cur_block if prev_block is None else prev_block)], axis=0)
        x = mid2_block.transpose(3,0,1,2).reshape(1,1280,12,10*25)
        x = np.transpose(x, (0,2,3,1))

        job = hailo_inference.run({
            'petrv2_repvggB0_transformer_pp_800x320/input_layer1': x,
            'petrv2_repvggB0_transformer_pp_800x320/input_layer2': matmul
        })

        # ★ 콜백이 넣어준 결과를 '직접' 꺼내서 out/meta를 같은 타이밍에 보냄
        try:
            outputs = local_out_q.get(timeout=1.0)   # dict: {name: [np.ndarray]}
            
        except pyqueue.Empty:
            continue

        pushed = False
        while not demo.get_terminate() and not pushed:
            try:
                queue_out.put(outputs, timeout=0.2)
                queue_meta_out.put(meta_data, timeout=0.2)
                pushed = True
            except pyqueue.Full:
                # 꽉 차면 살짝 쉬고 재시도
                time.sleep(0.01)

        prev_block = cur_block

        if demo.get_terminate():
            job.wait(100000)
            break