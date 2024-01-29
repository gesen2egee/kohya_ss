import os
import subprocess

def create_embs_directory():
    # 檢查當前目錄下是否存在名為 'embs' 的資料夾，如果不存在則創建
    if not os.path.exists('embs'):
        os.makedirs('embs')

def create_single_bat_file(path):
    commands = []  # 用於存儲所有命令的列表
    tag_names = []  # 存儲所有tag_name
    # 獲取當前執行腳本的目錄路徑
    current_path = os.getcwd().replace('\\', '/')
    
    for dir_name in os.listdir(path):
        full_dir_path = os.path.join(path, dir_name)

        if os.path.isdir(full_dir_path):
            if '_' in dir_name:
                # 取_之後,逗號之前的名字部分
                tag_name = dir_name.split('_')[-1].split(',')[0]
                if tag_name:
                    tag_name = tag_name.replace(' ', '_')
                    if '_' not in tag_name:
                        tag_name = tag_name + '^'
                    tag_names.append(tag_name)
                    if not os.path.exists(os.path.join('embs', f'{tag_name}.pt')):
                        command = f'python -m hcpdiff.tools.create_embedding deepghs/animefull-latest "{tag_name}" 4 --init_text *0.05'
                        commands.append(command)
    if not os.path.exists(os.path.join('embs', f'pivotal_tuning.pt')):
        command = f'python -m hcpdiff.tools.create_embedding deepghs/animefull-latest "pivotal_tuning" 4 --init_text *0.05'
        commands.append(command)

    # 寫入一個批處理檔案
    with open("run_commands.bat", "w") as bat_file:
        for cmd in commands:
            bat_file.write(cmd + "\n")
            
    try:
        subprocess.run('run_commands.bat', shell=True)
       
        os.remove('run_commands.bat')  # 執行完畢後刪除bat文件
    except Exception as e:
        print("執行批處理文件時出錯:", e)

    print('警告:如果emb過短(== 1 token)，進入kohyass會中斷，可以重命名一個長一點的，並且修改指令')
    print('創建一個無關的emb "pivotal_tuning"，因為目前觀察多個emb訓練可能會讓第一個emb損壞')
    print('修改emb的名字 因為當token跟原有token名相同時，會報錯。 另外將空格替換成_，因為會無法讀取，請將caption也做相同的修改') 
    print('訓練的emb概念有:') 
    for name in tag_names:
        print(name) 
    print('到kohya 貼上以下args:')    
    #print(final_command)
    
    # 生成並顯示最後的命令
    embeddings = ' '.join([f'"{current_path}/embs/{name}.pt"' for name in tag_names])
    final_command = f'--embeddings "{current_path}/embs/pivotal_tuning.pt" {embeddings} --continue_inversion --embedding_lr=0.01'
    print(final_command)  
    embeddings = ' '.join([f'"/workspace/npz/embs/{name}.pt"' for name in tag_names])    
    final_command = f'--embeddings "/workspace/npz/embs/pivotal_tuning.pt" {embeddings} --continue_inversion --embedding_lr=0.01'
#    print(final_command)

# 替換成您的資料夾路徑
path = '.'
create_embs_directory()
create_single_bat_file(path)
