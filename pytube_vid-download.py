import pytube


# link = "https://youtu.be/MNn9qKG2UFI"

# link  ='https://youtu.be/jFc91b3FJ7E'
link = 'https://youtu.be/WvhYuDvH17I'

yt = pytube.YouTube(link)
stream = yt.streams.get_highest_resolution()
stream.download()
