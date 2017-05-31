categories = {}
categories["artifact11"] = "http://imagenet.stanford.edu/api/text/imagenet.synset.geturls?wnid=n00021939"
categories["mics11"] = "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n13912260"

categories.keys.each do |key|
  Thread.new{
  `wget #{categories[key]}`
  `mkdir #{key}`
  f = File.new categories[key].split("/")[-1],"r"
  while not(f.eof?)
    img = f.readline()
    `cd #{key};wget #{img}`
  end
  }
end
