import requests
import os
from bs4 import BeautifulSoup

# 定义目标 URL 和保存目录
url = "https://class.damiaoedu.com:44313/codes/框架/新建文件夹/vue-wechat/"
save_dir = "D:/front end/框架/vue-wechat"

# 创建保存目录
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 发送请求并解析 HTML
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# 遍历所有链接并下载文件
for link in soup.find_all('a'):
    file_url = link.get('href')
    if file_url.endswith("/"):
        continue
    file_response = requests.get(url + file_url)
    file_path = os.path.join(save_dir, file_url)
    with open(file_path, "wb") as file:
        file.write(file_response.content)
    print(f"Downloaded: {file_url}")

print("Download complete.")