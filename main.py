import subprocess
import sys
import time
import json
import pymysql
import schedule


def load_config(filename):
    config = {}
    with open(filename, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config


def polling_db(config):
    try:
        db = pymysql.connect(**config['db'])
        with db.cursor() as cursor:
            sql = "SELECT * FROM cctv"
            cursor.execute(sql)
            results = cursor.fetchall()
            return results

    except:
        pass
    
    finally:
        cursor.close()
        db.close()


def manage_cctv(config, process_dict):
    print(f'process_dict: {process_dict}')
    results = polling_db(config)
    if results is not None:
        id_list = []
        for result in results:
            id_list.append(result[0])
            if result[0] not in process_dict:
                a = subprocess.Popen(args=[sys.executable, 'run.py', '--ipcam', f'rtsp://{result[4]}:{result[2]}@{result[3][7:]}', 
                                                                    '--socketport', f'{result[1]}', 
                                                                    '--cctvid', f'{result[0]}', 
                                                                    '--resturl', config['abnormal']['resturl'], 
                                                                    '--websocketurl', config['abnormal']['websocketurl'],
                                                                    '--posid', f'{result[6]}'])
                process_dict[result[0]] = a

        
        print(f'id_list: {id_list}')
        kill_id = []
        for process_id in process_dict:
            if process_id not in id_list:
                kill_id.append(process_id)

        for kill_i in kill_id:
            process_dict[kill_i].kill()
            print(f'kill {kill_i}')
            del(process_dict[process_id])


if __name__ == '__main__':
    process_dict = {}
    config = load_config('./config.json')

    

    schedule.every(5).seconds.do(manage_cctv, config, process_dict)

    while True:
        schedule.run_pending()
        time.sleep(1)


'''
a = subprocess.Popen(args=[sys.executable, 'run.py', '--ipcam', 'rtsp://icns:iloveicns@icns02.iptimecam.com:21244/stream_ch00_0', '--socketport', '30501', '--cctvid', '24'])
print(f'pid: {a.pid}')
process_dict[24] = a
print(process_dict)
del(process_dict[24])
print(process_dict)




time.sleep(20)
a.kill()

'''

