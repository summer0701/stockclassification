
import requests
import threading,time


def calli(name, low, high,interval):
    total = 0
    for i in range( low, high):
        try:
            r = requests.get(url = 'http://nis.gnu.ac.kr/orma/ht/om_ht_6071e.gnu?mode=doInsertTest')
        except:
            a =0
        time.sleep(interval)
        print(name + ' ' + str(r.json()))

    print("Subthread", total)


s = input('스레드개수: ')
ss = input('for 문 수 : ')
interval = input('시간: ')
for i in range(0,int(s)):
    threading.Thread(target=calli, args=('t'+str(i),0, int(ss), float(interval))).start()


