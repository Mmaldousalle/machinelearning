function img=lex2img(lex)

s_img=sqrt(size(lex,1));
img=zeros(s_img,s_img);

for i=1:s_img
   img(i,:)=lex(((i-1)*s_img)+1:i*s_img); 
end