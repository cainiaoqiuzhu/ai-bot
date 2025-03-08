# encoding:utf-8
import urllib, urllib3, sys, uuid
import ssl
from config import weather_code

host = 'https://ali-weather.showapi.com'
path = '/weatherhistory'
method = 'GET'
appcode = weather_code['appcode']
querys = 'areaCode=areaCode&area=%E4%B8%BD%E6%B1%9F&month=201601&startDate=20160504&endDate=20160510'
bodys = {}
url = host + path + '?' + querys

http = urllib3.PoolManager()
headers = {
    'Authorization': 'APPCODE ' + appcode
}
response = http.request('GET', url, headers=headers)
content = response.data.decode('utf-8')
if (content):
    print(content)

