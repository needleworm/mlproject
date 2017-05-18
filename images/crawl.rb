total = 2000
dup = []
n = 0
while n < total
  jp = "#{n}.jpeg" 
  cmd = "wget https://source.unsplash.com/random/1024x1024 -O #{jp}"
  `#{cmd}`
  md5 = `md5 #{jp}`.split[-1]
  if not dup.include? md5
    dup.push md5
    n += 1
  end
end
