hello
this script i threw together is basically for training LoRAs without all the annoying crap that makes you wanna rage quit.  
got sick of massive UIs or waiting forever for other tools to do the simplest shit. now u just dump your pics in a folder, run the thing, pick from the menu, and boom.

it handles a couple things pretty decently:
- caches the latents so you don't re-encode images every single time (huge time saver)
- grabs frames from videos too (up to 10 per clip) if you got opencv installed
- two training modes: 
  normal one is basic – single rank + alpha, easy mode
  finetune mode lets you tweak separate learning rates and ranks for unet vs text encoder (sometimes that actually helps a lot)
- auto-resumes if it crashes or you kill it, so progress doesn't vanish
- shows a live VRAM bar so you can see when you're about to cook your card

why i even made this  
kohya_ss is solid but feels like using a tank to go to the corner store when i just wanna train one character or style quick.  
onetrianer is cool but the setup makes me wanna die.  
this one is way simpler: folder → menu → train → maybe generate some test pics with what u just made.

how to actually use it  
1. throw your images (.txt caption files if u bothered) into a folder  
2. python run.py  
3. pick 1 for normal training, 3 for finetune mode, 4 to generate images  
4. keep batch size low and res at 512 unless you got monster VRAM 24gb, otherwise itll choke

what i run it on  
python 3.10 or 3.11  
torch with cuda  
pulls in diffusers, transformers, peft, accelerate etc and installs whatever's missing first run  
opencv-python if you want video support (optional)

flies on 3090 and 4090 gpus. 8gb cards can do it at 512 res low batch but itll crawl like hell.

janky stuff i know about  
- progress bar sometimes freaks out in windows cmd prefferablt use windows terminal or my favourite, powershell instead 
- super high steps like 200 can make images look weird sometimes probably the scheduler being dumb, not my code  
- no negative prompts, controlnet, ip-adapter or any of that fancy stuff yet maybe one day, idk

if you make something cool, cursed, or straight-up weird with it, drop it here or dm me, i wanna see.  
if it explodes just tell me what broke and i'll probably fix it eventually.(Side note this ran 51k images on 16k steps with another 51k captions in just 6 hours i believe idk i feel asleep)

— me
u are free to modify it but do not distribute it without any credits or without me knowing 
