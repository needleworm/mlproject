categories = {}
categories["artifact"] = "http://imagenet.stanford.edu/api/text/imagenet.synset.geturls?wnid=n00021939"

categories.keys.each do |key|
  #`wget #{url}`
  `mkdir #{key}`

  f = File.new categories[key].split("/")[-1],"r"
  while not(f.eof?)
    img = f.readline()
    `cd #{key};wget #{img}`
  end
end
