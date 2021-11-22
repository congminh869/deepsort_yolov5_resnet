# # importing the module 
# from pytube import YouTube 
  
# # where to save 
# SAVE_PATH = ".data/" #to_do 
  
# # link of the video to be downloaded 
# link="https://www.youtube.com/watch?v=u9Zciw-IsEA"
  
# try: 
#     # object creation using YouTube
#     # which was imported in the beginning 
#     yt = YouTube(link) 
# except: 
#     print("Connection Error") #to handle exception 
# yt = YouTube(link)
# # filters out all the files with "mp4" extension 
# mp4files = yt.filter('mp4') 
  
# #to set the name of the file
# yt.set_filename('GeeksforGeeks Video')  
  
# # get the video with the extension and
# # resolution passed in the get() function 
# d_video = yt.get(mp4files[-1].extension,mp4files[-1].resolution) 
# try: 
#     # downloading the video 
#     d_video.download(SAVE_PATH) 
# except: 
#     print("Some Error!") 
# print('Task Completed!') 

from pytube import Playlist
play_list = []

#Accessing the Playlist
playlist = Playlist('https://www.youtube.com/playlist?list=PLIy896S2zPw4J58kVUgvwgyrH9xHf2q7d')

#Checking the number of videos in the playlist
print('Number of videos in playlist: %s' % len(playlist.video_urls))

#Parsing through the playlist and appending the video urls in play_list
for video_url in playlist.video_urls:
    print(video_url)
    play_list.append(video_url)

#Looping through the list
for i in play_list:
  try:
    yt = YouTube(i)
    print('Downloading Link: ' + i)
    print('Downloading video: ' + yt.streams[0].title)
  except:
    print("Connection Error")
	
	#filters out all the files with "mp4" extension
  stream = yt.streams.filter(res="360p").first()
  stream.download("data/")
print('Task Completed!')
