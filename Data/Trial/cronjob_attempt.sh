cd /home/rohit/5/SMiAI/Project/Code/Data/Trial/
date=`date +%d_%m`
if [ ! -e $date ]
then
	mkdir "$date"
fi
# date >> /home/rohit/5/SMiAI/Project/Code/Data/Trial/cronjob_attempt.txt
