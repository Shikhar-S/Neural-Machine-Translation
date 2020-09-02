sudo cp mymodule.ko /lib/modules/$(uname -r)/kernel/drivers/
cat /boot/config-`uname -r` | grep IP_MROUTE
cat /boot/config-`uname -r`
find /lib/modules/`uname -r` -regex .*perf.*
grep “HIGHMEM” /boot/config-`uname -r`
cat /proc/2671/maps | grep `which tail`
grep PROBES /boot/config-$(uname -r)
grep UTRACE /boot/config-$(uname -r)
grep ds1337 /lib/modules/`uname -r`/modules.alias
sed -i "s/\\\\\n//g" filename
set -e
find . -name \*.py -print0 | xargs -0 sed -i '1a Line of text here'
find . -name \*.py | xargs sed -i '1a Line of text here'
find ~ -type d -exec chmod +x {} \;
rename 's/(.*)$/new.$1/' original.filename
rename 's/^/new./' original.filename
nl -s prefix file.txt | cut -c7-
nl -s "prefix_" a.txt | cut -c7-
find /volume1/uploads -name "*.mkv" -exec mv \{\} \{\}.avi \;
cat <(crontab -l) <(echo "1 2 3 4 5 scripty.sh") | crontab -
ping google.com | xargs -L 1 -I '{}' date '+%c: {}'
nl -ba infile
nl -ba long-file \
echo "$string" | nl -ba -s') '
crontab -l -u user | cat - filename | crontab -u user -
cat file1 file2 | crontab
crontab filename
pushd "$HOME/Pictures"
sudo chmod +x java_ee_sdk-6u2-jdk-linux-x64.sh
chmod +x pretty-print
chmod +x rr.sh
chmod a+x ComputeDate col printdirections
chmod +x *.sh
chmod g+w $(ls -1a | grep -v '^..$')
chmod g+w .[^.]* ..?*
find . -maxdepth 0 -type f -exec chmod g+w {} ';'
chmod g+w * ...*
nl -v1000001 file
sed 's/3d3d/\n&/2g' temp | split -dl1 - temp
nl -s"^M${LOGFILE}>  "
sudo chmod +rx $(which node)
find . -type d -exec chmod +rx {} \;
find . -name "rc.conf" -exec chmod o+r '{}' \;
find . -type f -iname '*.txt' -print0 | xargs -0 mv {} {}.abc
find ~/dir_data -type d -exec chmod a+xr,u+w {} \;
v=5 env|less
TESTVAR=bbb env | fgrep TESTVAR
pushd %Pathname%
touch -d "$(date -r filename) - 2 hours" filename
touch -d "$(date -r "$filename") - 2 hours" "$filename"
chmod a+x myscript.sh
chmod a+x $pathToShell"myShell.sh"
sudo chmod u+s `which Xvfb`
yes n | rm -ir dir1 dir2 dir3
yes | cp * /tmp
yes | rm -ri foo
yes y | rm -ir dir1 dir2 dir3
sed 's/.*/& Bytes/' "$TEMPFILE" | column -t
find -type f | xargs -I {} mv {} {}.txt
echo -e "1\n2\n3" | sed 's/.*$/&<br\/>/'
sed 's/$/\r/g' input |od -c
awk 'NR==1 {print $0, "foo", "bar"; next} {print $0, ($2=="x"?"-":"x"), ($4=="x"?"-":"x")}' file | column -t
find . -type f -name "*.java" | xargs tar rvf myfile.tar
find . -name -type f '*.mp3' -mtime -180 -print0 | xargs -0 tar rvf music.tar
find . \( -iname "*.png" -o -iname "*.jpg" \) -print -exec tar -rf images.tar {} \;
find . -mtime -1 -type f -exec tar rvf "$archive.tar" '{}' \;
find . -mtime -1 -type f -print0 | xargs -0 tar rvf "$archive.tar"
history -a
history -r .cwdhist
history -r file.txt
LOGNAME="`basename "$0"`_`date "+%Y%m%d_%H%M"`"
name="$(date +'%d%m%Y-%H-%M')_$(whoami)"
LBUFFER+="$(date)"
PROMPT_COMMAND='echo "$(date +"%Y/%m/%d (%H:%M)") $(history 1 |cut -c 7-)" >> /tmp/trace'
KEY+=`date -r "$arg" +\ %s`
find . -name text.txt | sed 's|.*/\(.*\)/.*|sed -i "s@^@\1 @" & |' | sh
rsync -rvz -e 'ssh -p 2222' --progress ./dir user@host:/path
rsync -av --copy-dirlinks --delete ../htmlguide ~/src/
rsync -avh /home/abc/* /mnt/windowsabc
rsync -a --stats --progress --delete /home/path server:path
rsync -av /home/user1 wobgalaxy02:/home/user1
rsync -avz --progress local/path/some_file usr@server.com:"/some/path/"
rsync -avzru --delete-excluded server:/media/10001/music/ /media/Incoming/music/
rsync -avzru --delete-excluded /media/Incoming/music/ server:/media/10001/music/
rsync -av --exclude '*.svn' user@server:/my/dir .
rsync -avv source_host:path/to/application.ini ./application.ini
rsync -chavzP --stats user@remote.host:/path/to/copy /path/to/local/storage
rsync -chavzP --stats /path/to/copy user@host.remoted.from:/path/to/local/storage
rsync -avlzp user@remotemachine:/path/to/files /path/to/this/folder
rsync -av --rsync-path="sudo rsync" /path/to/files user@targethost:/path
rsync -av /path/to/files user@targethost:/path
rsync -azP -e "ssh -p 2121" /path/to/files/source user@remoteip:/path/to/files/destination
rsync -avlzp /path/to/sfolder name@remote.server:/path/to/remote/dfolder
rsync -aHvz /path/to/sfolder name@remote.server:/path/to/remote/dfolder
rsync -aHvz /path/to/sfolder/ name@remote.server:/path/to/remote/dfolder
rsync -avz --ignore-existing /source folder/* user@remoteserver:/dstfolder/
rsync -ravz /source/backup /destination
rsync -a --relative /top/a/b/c/d remote:/
rsync --progress -avhe ssh /usr/local/  XXX.XXX.XXX.XXX:/BackUp/usr/local/
rsync -rave "ssh -i /home/test/pkey_new.pem" /var/www/test/ ubuntu@231.210.24.48:/var/www/test
rsync -aqz _vim/ ~/.vim
rsync -aqz _vimrc ~/.vimrc
rsync -a --delete blanktest/ test/
rsync -aPSHiv remote:directory .
rsync -ave ssh fileToCopy ssh.myhost.net:/some/nonExisting/dirToCopyTO
rsync -avR foo/bar/baz.c remote:/tmp/
rsync -a myfile /foo/bar/
rsync -vuar --delete-after path/subfolder/ path/
rsync -a --exclude .svn path/to/working/copy path/to/export
rsync -avR somedir/./foo/bar/baz.c remote:/tmp/
rsync -azP -e "ssh -p PORT_NUMBER" source destination
rsync -a -v src dst
rsync -a -v --ignore-existing src dst
rsync -av --delete src-dir remote-user@remote-host:dest-dir
rsync -avz foo:src/bar /data/tmp
rsync -aP --include=*/ --include=*.txt --exclude=* . /path/to/dest
find . -name \*.xml | grep -v /workspace/ | tr '\n' '\0' | xargs -0 tar -cf xml.tar
find . -type f -name "*html" | xargs tar cvf htmlfiles.tar -
find /path/to/directory/* -maxdepth 0 -type d -printf "%P\n" -exec sudo tar -zcpvf {}.tar.gz {} \;
find /path/* -maxdepth 0 -type d -exec sudo tar -zcpvf {}.tar.gz {} \;
find data/ -name 'filepattern-*2009*' -exec tar uf 2009.tar '{}' +
find data/ -name filepattern-*2009* -exec tar uf 2009.tar {} ;
find data/ -name filepattern-*2009* -print0 | xargs -0 tar uf 2009.tar
rsync -a -f"+ info.txt" -f"+ data.zip" -f'-! */' folder1/ copy_of_folder1/
rsync -vaut ~/.env* ~/.bash* app1:
rsync -av --files-from=- --rsync-path="sudo rsync" /path/to/files user@targethost:/path
rsync -av remote_host:'$(find logs -type f -ctime -1)' local_dir
rsync -auve "ssh -p 2222" . me@localhost:/some/path
rsync -av . server2::sharename/B
rsync -az --delete /mnt/data/ /media/WD_Disk_1/current_working_data/;
rsync symdir/ symdir_output/ -a --copy-links -v
rsync -avz tata/ tata2/
rsync -avR $i /iscsi;
rsync -av $myFolder .
bzip2 -c file | tee -a logfile
rsync -a --filter="-! */" sorce_dir/ target_dir/
rsync -a /mnt/source-tmp /media/destination/
sudo rsync -az user@10.1.1.2:/var/www/ /var/www/
rsync -av --progress sourcefolder /destinationfolder --exclude thefoldertoexclude
rsync -av --progress --exclude=*.VOB --exclude=*.avi --exclude=*.mkv --exclude=*.ts --exclude=*.mpg --exclude=*.iso --exclude=*ar --exclude=*.vob --exclude=*.BUP --exclude=*.cdi --exclude=*.ISO --exclude=*.shn --exclude=*.MPG --exclude=*.AVI --exclude=*.DAT --exclude=*.img --exclude=*.nrg --exclude=*.cdr --exclude=*.bin --exclude=*.MOV --exclude=*.goutputs* --exclude=*.flv --exclude=*.mov --exclude=*.m2ts --exclude=*.cdg --exclude=*.IFO --exclude=*.asf --exclude=*.ite /media/2TB\ Data/data/music/* /media/wd/network_sync/music/
find / -print0 | xargs -0 tar cjf tarfile.tar.bz2
tar -czf /fss/fi/outfile.tar.gz `find /fss/fin -d 1 -type d -name "*" -print`
sudo crontab -e -u apache
find . -type f -print0 | xargs -0 chmod 644
find . -type d -print0 | xargs -0 chmod 755
ifconfig eth0 hw ether 00:80:48:BA:d1:30
scp -p /home/reportuser/dailyReport.doc root@localhost:/root/dailyReports/20150105/
scp -o StrictHostKeyChecking=no root@IP:/root/K
scp -rp "DAILY_TEST_FOLDER" "root@${IPADDRESS}:/home/root/"
find /etc -name "*.txt" | xargs -I {} mv {} {}.bak
find /etc -print0 -name "*.txt" | xargs -I {} -0 mv {} {}.bak
find -name "*.php" –exec cp {} {}.bak \;
find . -name "*.java" -exec cp {} {}.bk \;
ifconfig eth0 down
find /usr/local/svn/repos/ -maxdepth 1 -mindepth 1 -type d -printf "%f\0" | xargs -0 -I{} echo svnadmin hotcopy /usr/local/svn/repos/\{\} /usr/local/backup/\{\}
md5sum *.java | awk '{print $1}' | sort | uniq -d
find . -type f -exec md5sum \{\} \;
find . | xargs md5sum
FILE="/tmp/command_cache.`echo -n "$KEY" | md5sum | cut -c -10`"
md5=$(echo "$line"|md5sum)
checksum=`md5sum /etc/localtime | cut -d' ' -f1`
ls -alR -I dev -I run -I sys -I tmp -I proc /path | md5sum -c /tmp/file
cpio -i -e theDirname | md5sum
echo -n "" | md5sum
echo -n | md5sum
md5sum "$ecriv"
md5=$(md5sum $item | cut -f1 -d\ )
md5="$(md5sum "${my_iso_file}")"
md5=`md5sum ${my_iso_file} | cut -b-32`
md5sum "$source_file" "$dest_file"
find "$path" -type f -print0 | sort -z | xargs -r0 md5sum | md5sum
md5sum main.cpp*
SUM=$(tree | md5sum)
echo "a" | md5sum
echo -n 'exampleString' | md5sum
echo -n "logdir" | md5sum - | awk '{print $1}'
echo "password" | md5sum
echo -n "yourstring" |md5sum
grep -ar -e . --include="*.py" /your/dir | md5sum | cut -c-32
cat *.py | md5sum
grep -ar -e . /your/dir | md5sum | cut -c-32
grep -aR -e . /your/dir | md5sum | cut -c-32
find -maxdepth 1 -type f -exec md5sum {} \; | sed 's/[^(]*(\([^)]*\)) =/\1/'
find -maxdepth 1 -type f -exec md5sum {} \; | awk '{s=$2; $2=$1; $1=s;}1'
ls -p | grep -v / | xargs md5sum | awk '{print $2,$1}'
find . -name '.svn' -prune -o -type f -printf '%m%c%p' | md5sum
find /path -type f -name "*.py" -exec md5sum "{}" +;
echo -n -e '\x61' | md5sum
cat $FILES | md5sum
find /path -type f | sort -u | xargs cat | md5sum
cat $(echo $FILES | sort) | md5sum
md5sum filename |cut -f 1 -d " "
find . -maxdepth 1 -type f | md5sum
find "$path" -type f -print0 | sort -z | xargs -r0 md5sum | md5sum
du -csxb /path | md5sum -c file
find /path/to/dir/ -type f -name *.py -exec md5sum {} + | awk '{print $1}' | sort | md5sum
tar c dir | md5sum
find -iname "MyCProgram.c" -exec md5sum {} \;
find /path/to/dir/ -type f -name "*.py" -exec md5sum {} + | awk '{print $1}' | sort | md5sum
md5sum $(which cc)
md5sum $(which gcc)
md5sum `which c++`
find "$PWD" / -iname '*.jpg' -exec du -s {} + | sed "s/^/$(hostname): /"
find . -name "*jpg" -exec du -k {} \; | awk '{ total += $1 } END { print total/1024 " Mb total" }'
depth=$(pstree -sA $processid | head -n1 | sed -e 's#-+-.*#---foobar#' -e 's#---*#\n#g' -eq | wc -l)
env | grep -i shell
cat report.txt | grep -i error | more
rename -v 's/\.JPG/\.jpeg/' *.JPG
crontab -l | sed '/anm\.sh/s#\/5#\/10#' | crontab -
crontab -l | sed '/anm\.sh/s,^\*/5,*/10,' | crontab -
sudo find ./bootstrap/cache/ -type d -exec chown apache:laravel {} \;
sudo find ./storage/ -type d -exec chown apache:laravel {} \;
find htdocs -type f -exec chmod 664 {} + -o -type d -exec chmod 775 {} +
find ~ -group vboxusers -exec chown kent:kent {} \;
find . \( \! -user xx -exec chown -- xx '{}' + -false \)
sudo find /var/www -nouser -exec chown root:apache {} \;
cd -P "$dir1"
cd /lib/modules/$(uname -r)/
cd  /path/to/pdf
cd -L ..
cd $(dirname $(dirname $(which perl)))/lib
cd "$(find . -print0 | sort -z | tr '\0' '\n' | tail -1)"
cd $(basename $1 .tar.gz)
cd /home/`whoami`
cd "$(dirname "$1")"
cd "$(dirname $(which oracle))"
cd $(dirname $(which oracle))
cd $(dirname `which oracle`)
cd $(dirname $(which $0) )
cd $(which oracle | xargs dirname)
cd "$(grep DOWNLOAD $HOME/.config/user-dirs.dirs | cut -f 2 -d "=" | tr "\"" "\n" | tr -d "\n")"
cd "$(grep DOWNLOAD $HOME/.config/user-dirs.dirs | cut -f 2 -d "=" | tr "\"" "\n")"
cd $( ~/marker.sh go "$@" )
cd $(dirname $(readlink -f $0))
cd $(readlink /proc/$PID/cwd)
cd $(readlink -f $(dirname $0))
cd
sudo  find /var/www/html/ -type d -exec chmod 775 {} \;
sudo  find /var/www/html/ -type f -exec chmod 664 {} \;
find . -name "*.css" -exec sed -i -r 's/#(FF0000|F00)\b/#0F0/' {} \;
chown -v root:root /path/to/yourapp
find /path/to/directory -type f -exec chmod 644 {} +
cd $(dirname $(readlink -f $0))
find / -group 2000 -exec chgrp -h foo {} \;
find . -name '*.php' -exec chmod 755 {} \; | tee logfile.txt
chown user_name file
sudo chown root:wheel com.xxxx.adbind.plist
chown root:root script.sh
chown user_name folder
sudo chown el my_test_expect.exp
chown $1:httpd .htaccess
chown $FUID:$FGID "$FILE2"
chown -- "$user:$group" "$file"
sudo chown bob:sftponly /home/bob/writable
sudo chown root:dockerroot /var/run/docker.sock
sudo chown root:wheel adbind.bash
sudo chown root:wheel bin
sudo chown root:www-data /foobar/test_file
sudo chown `whoami` /data/db
sudo chown `whoami` /vol
find /path/to/look/in/ -type d -name '.texturedata' -exec chmod 000 {} \; -prune
find /path/to/look/in/ -type d -name '.texturedata' -prune -print0 | xargs -0 chmod 000
find "$d/" -type d -print0 | xargs -0 chmod 755
find -perm 777 | xargs -I@ sudo chmod 755 '@'
find . -name "*.php" -exec chmod 755 {} \;
find . -name "*.php" -exec chmod 755 {} + -printf '.' | wc -c
find . -name "*.php" -exec chmod 755 {} \; -exec /bin/echo {} \; | wc -l
chmod 444 .bash_logout .bashrc .profile
sudo chmod 755 .git/hooks/pre-commit
sudo chmod 777 .git/hooks/prepare-commit-msg
sudo chmod 755 /dvtcolorconvert.rb
chmod 777 /usr/bin/wget
sudo chmod 755 mksdcard
find . -type d -exec chmod 755 {} +
find ~/dir_data -type d -exec chmod a+xr,u+w {} \;
find ./debian -type d | xargs chmod 755
find . -name "*.php" -exec chmod 755 {} + -printf '.' | wc -c
find . -name '*.php' -exec chmod 755 {} \; -exec echo '+' \;
find . -name "*.php" -exec chmod 755 {} \; -exec /bin/echo {} \; | wc -l
find . -type f -exec chmod 644 {} +
find ~/dir_data -type f -exec chmod a-x,u+w {} \;
chmod 555 /home/sshtunnel/
find /path -type d -exec chmod 0755 "{}" \;
find /path -type d -exec chmod 0755 {} \;
find /path -type d | xargs chmod 0755
find . -type f -exec chmod 500 {} ';'
find . -name "*.rb" -type f -exec chmod 600 {} \;
find /usr/local -name "*.html" -type f -exec chmod 644 {} \;
find /path/to/someDirectory -type d -print0 | xargs -0 sudo chmod 755
find . -type f | xargs -I{} chmod -v 644 {}
find . -type f | xargs chmod -v 644
find ./ -type f -print0 | xargs -t -0 chmod -v 644
find . -type f -print | sed -e 's/^/"/' -e 's/$/"/' | xargs chmod 644
find /path/to/someDirectory -type f -print0 | xargs -0 sudo chmod 644
find /path/to/dir/ -type f -print0 | xargs -0 chmod 644
find /path/to/dir ! -perm 0644 -exec chmod 0644 {} \;
find /path/to/dir/ -type f ! -perm 0644 -print0 | xargs -0 chmod 644
find . -type d -print0|xargs -0 chmod 644
find . -perm 755 -exec chmod 644 {} \;
find . -type f -perm 755 -exec chmod 644 {} \;
find . -type f -name '*.php' -exec chmod 644 {} \;
find . -type f -exec chmod 644 {} \;
find . -mindepth 1 -type d | xargs chmod 700
find . -mindepth 2 | xargs chmod 700
find /path/to/dir -type d -exec chmod 755 {} \;
find . -type d | xargs chmod -v 755
find . -type d -print | sed -e 's/^/"/' -e 's/$/"/' | xargs chmod 755
find . -type d -exec chmod 755 {} \;
find . -type d -exec chmod 777 {} \;
find . -type f -exec chmod u=rw,g=r,o= '{}' \;
find . -type f -exec chmod u=rw,g=r,o= '{}' \;
find . -type d -exec chmod u=rwx,g=rx,o= '{}' \;
find . -type d -exec chmod u=rwx,g=rx,o= '{}' \;
find htdocs cgi-bin -name "*.cgi" -type f -exec chmod 755 {} \;
find . -type f -exec sed -i 's/searc/replace/g' {} \;
cp --remove-destination $(readlink $f) $f
find . -name "*.txt" | sed "s/\.txt$//" | xargs -i echo mv {}.txt {}.bak | sh
chown :friends myfile
find . -type d | sed -e 's/\.\///g' -e 's/\./avoid/g' | grep -v avoid | awk '{print $1"\t"$1}' | xargs chgrp
find . -type d | sed -e 's/\.\///g' | awk '{print $1, $1}' | xargs chgrp
find . -group root -print | xargs chgrp temp
chown root:root it
sudo chown root:root testfile.txt
sudo chown root:root uid_demo
chown $JBOSS_USER $JBOSS_CONSOLE_LOG
sudo chown nobody /var/www/html/mysite/images/
sudo chown nobody /var/www/html/mysite/tmp_file_upload/
chown user destination_dir
sudo chown root process
find /mydir -type f -name "*.txt" -execdir chown root {} ';'
ls /empty_dir/ | xargs -L10 chown root
ls /empty_dir/ | xargs -n10 chown root
find . -not -iwholename './var/foo*' -exec chown www-data '{}' \;
find dir_to_start -name dir_to_exclude -prune -o -print0 | xargs -0 chown owner
find dir_to_start -not -name "file_to_exclude"  -print0 | xargs -0 chown owner
sudo chown hduser:hadoop {directory path}
chown owner:nobody public_html
chown root:specialusers dir1
chown user:group file ...
sudo chown root. /etc/udev/rules.d/51-android.rules
sudo chown root /home/bob
sudo chown root file.sh
find . -user aluno1 -exec chown aluno2 {}
find -user root -exec chown www-data {} \;
find . -exec chown myuser:a-common-group-name {} +
find -x / -user george -print0 | xargs -0 chown eva
find . -type d -user harry -exec chown daisy {} \;
find . -type f -exec chmod 644 {} \;
find . -type f -exec chmod 0644 {} +
find . -type f -exec chmod 0644 {} \;
find . -type d -exec chmod 0755 {} \;
find . -type f | xargs -I{} chmod -v 644 {}
find . -type f | xargs chmod -v 644
find . -type d | xargs chmod -v 755
find /var/ftp/mp3 -name '*.mp3' -type f -exec chmod 644 {} \;
find . -maxdepth 1 -type d -exec chmod -R 700 {} \;
find . -type d -exec chmod 755 {} \;
touch -h somesymlink
find /var/www -print0 | xargs -0 chown www-data:www-data
find . -type d -user harry -exec chown daisy {} \;
cd foo | cat
cd -P xyz
cd `cat $HOME/.lastdir`
cd "$(dirname "$(which oracle)")"
cd "$(dirname $(which oracle))"
cd $(dirname $(which oracle))
cd $(dirname `which oracle`)
cd $(which oracle | xargs dirname)
cd `dirname $TARGET_FILE`
cd -P ..
cd "$(dirname "$(which oracle)")"
cd `dirname $(which python)`
cd "$TAG"
find / -user 1005 -exec chown -h foo {} \;
chown amzadm.root  /usr/bin/aws
chgrp btsync /etc/btsync/[prefered conf name].conf
chgrp www-data /home/www-user/php_user.sh
chgrp forge /var/run/fcgiwrap.socket
chgrp loggroup logdir
chgrp groupb myprog
chgrp "${USER}" myprogram
chgrp god public private
chgrp pub public
chgrp Workers shared
chgrp target_group target_directory
sudo chgrp gpio /sys/class/gpio/export /sys/class/gpio/unexport
chgrp
cd $(dirname $(which ssh));
find . -type f -iname "*.txt" -print | xargs grep "needle"
find . -type f -iname "*.txt" -print0 | xargs -0 grep "needle"
od -t x2 -N 1000 $file | cut -c8- | egrep -m1 -q ' 0d| 0d|0d$'
mount -l | grep 'type nfs' | sed 's/.* on \([^ ]*\) .*/\1/' | grep /path/to/dir
AMV=$(mount -l | grep "\[$VLABEL\]")
mount | grep -q ~/mnt/sdc1
df $path_in_question | grep " $path_in_question$"
is_nullglob=$( shopt -s | egrep -i '*nullglob' )
mount |grep nfs
mount | grep $(readlink -f /dev/disk/by-uuid/$UUID )
cat *.txt | sort | sort -u -c
pstree --show-parents -p $$ | head -n 1 | sed 's/\(.*\)+.*/\1/' | wc -l
uname -m | grep '64'
find "`echo "$some_dir"`" -maxdepth 0 -empty
kill -0 1
find "$somedir" -maxdepth 0 -empty -exec echo {} is empty. \;
comm -23 <(sort subset | uniq) <(sort set | uniq) | head -1
find "$somedir" -type f -exec echo Found unexpected file {} \;
ls `readlink somelink`
du -csxb /path | md5sum -c file
ssh -S my-ctrl-socket -O check jm@sampledomain.com
ssh -O check officefirewall
sudo env
df $path_in_question | grep " $path_in_question$"
df /full/path | grep -q /full/path
pstree --show-parents -p $$ | head -n 1 | sed 's/\(.*\)+.*/\1/' | grep screen | wc -l
bzip2 -t file.bz2
groups monit |grep www-data
find . \( -name a.out -o -name '*.o' -o -name 'core' \) -exec rm {} \;
find . -type d -name ".svn" -print | xargs rm -rf
kill -9 $(ps -A -ostat,ppid | grep -e '[zZ]'| awk '{ print $2 }')
kill $(ps -A -ostat,ppid | awk '/[zZ]/{print $2}')
history -c
history -cr
history -c
echo `clear`
clear
ssh -S my-ctrl-socket -O exit jm@sampledomain.com
dir="`echo $dir | sed s,//,/,g`"
paste -d "" - -
diff "$source_file" "$dest_file"
diff current.log previous.log | grep ">\|<" #comparring users lists
diff -up fastcgi_params fastcgi.conf
diff -u file1 file2
find . -name "*.csv" -exec diff {} /some/other/path/{} ";" -print
find . -okdir diff {} /some/other/path/{} ";"
awk 'NR==1 { print; next } { print $0, ($1 == a && $2 == b) ? "equal" : "not_equal"; a = $1; b = $3 }' file | column -t
find . -name *.xml -exec diff {} /destination/dir/2/{} \;
diff -ENwbur repos1/ repos2/
diff -u A1 A2 | grep -E "^\+"
comm abc def
diff -Naur dir1/ dir2
diff -Nar /tmp/dir1 /tmp/dir2/
comm <(sort -n f1.txt) <(sort -n f2.txt)
comm <(sort f1.txt) <(sort f2.txt)
diff <(echo hello) <(echo goodbye)
diff <(ls /bin) <(ls /usr/bin)
diff <(zcat file1.gz) <(zcat file2.gz)
find FOLDER1 -type f -print0 | xargs -0 -I % find FOLDER2 -type f -exec diff -qs --from-file="%" '{}' \+
comm -23 <(ls) <(ls *Music*)
DST=`dirname "${SRC}"`/`basename "${SRC}" | tr '[A-Z]' '[a-z]'`
g=`dirname "$f"`/`basename "$f" | tr '[A-Z]' '[a-z]'`
pstree -p | grep git
gzip archive.tar
hey=$(echo "hello world" | gzip -cf)
gzip -c my_large_file | split -b 1024MiB - myfile_split.gz_
gzip "$file"
gzip "{}"
gzip */*.txt
find . -type f -name "*.txt" -exec gzip {} \;
find ./ -name "*.img" -exec bzip2 -v {} \;
find $LOGDIR -type d -mtime +0 -exec compress -r {} \;
find $LOGDIR -type d -mtime -1 -exec compress -r {} \;
find $PATH_TO_LOGS -maxdepth 1 -mtime +$SOME_NUMBER_OF_DAYS -exec gzip -N {} \;
find $FILE -type f -mtime 30 -exec gzip {} \;
find $FILE -type f -not -name '*.gz' -mtime 30 -exec gzip {} \;
find /source -type f -print0 | xargs -0 -n 1 -P $CORES gzip -9
find . -type f -print0 | xargs -0r gzip
echo *.txt | xargs gzip -9
sudo find / -xdev -type f -size +100000 -name "*.log" -exec gzip -v {} \;
sudo find / -xdev -type f -size +100000 -name "*.log" -exec gzip {} \; -exec echo {} \;
gzip -k *cache.html
find . -type f -name "*cache.html" -exec gzip -k {} \;
find folder -type f -exec gzip -9 {} \; -exec mv {}.gz {} \;
find . \! -name "*.Z" -exec compress -f {} \;
echo gzip. $( gzip | wc -c )
gzip
find . -type f  -mtime +7 | tee compressedP.list | xargs -I{} -P10 compress {} &
find . -type f  -mtime +7 | tee compressedP.list | xargs compress
bzip2 file | tee -a logfile
find -name \*.xml -print0 | xargs -0 -n 1 -P 3 bzip2
bzip2 *
find PATH_TO_FOLDER -maxdepth 1 -type f -exec bzip2 -zk {} \;
compress $* &
bzip2 -k example.log
find "$1" -type f | egrep -v '\.bz2' | xargs bzip2 -9 &
find ~/ -name '*.txt' -print0 | xargs -0 wc -w | awk 'END { print $1/(NR-1) }'
find ~/Journalism  -name '*.txt' -print0 | xargs -0 wc -w | awk '$1 < 2000 {v += $1; c++} END {print v/c}'
find . -iname '*test*' -exec cat {} \;
find . -name '*test*' -exec cat {} \;
paste -s -d' \n' input.txt
scp -qv $USER@$HOST:$SRC $DEST
ssh -S "$SSHSOCKET" -O exit "$USER_AT_HOST"
ssh -M -f -N -o ControlPath="$SSHSOCKET" "$USER_AT_HOST"
ssh -l ${USERNAME} ${HOSTNAME} "${SCRIPT}"
scp -v user@remotehost:/location/KMST_DataFile_*.kms
scp -v /my_folder/my_file.xml user@server_b:/my_new_folder/
ssh -o UserKnownHostsFile=/dev/null username@hostname
scp -P 1234 user@[ip address or host name]:/var/www/mywebsite/dumps/* /var/www/myNewPathOnCurrentLocalMachine
scp -P 2222 /absolute_path/source-folder/some-file user@example.com:/absolute_path/destination-folder
scp -c blowfish -r user@your.server.example.com:/path/to/foo /home/user/Desktop/
find . -name "*.txt" \( -exec echo {} \; -o -exec true \; \) -exec grep banana {} \;
yes | mv ...
yes a=\"20131202\" | sed -e :a -e 's/...\([0-9]\{4\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)./\1 \2 \3/'
ping google.com | awk -F'[ =]' 'NR>1{print system("echo -n $(date +%s)"), $11}'
yes | cat | more
yes | rm
ping -b 10.10.0.255 | grep 'bytes from' | awk '{ print $4 }'
ping -c1 1199092913 | head -n1 | grep -Eow "[0-9]+[.][0-9]+[.][0-9]+[.][0-9]+"
echo 595a | awk -niord '$0=chr("0x"RT)' RS=.. ORS= | od -tx1c
echo "luke;yoda;leila" | tr ";" "\n"
echo abc | od -A n -v -t x1 | tr -d ' \n'
find -type f -name '*.au' | awk '{printf "sox %s %s\n",$0,$0".wav" }' | bash
cal -h 02 2012| cut -c4-17 | sed -r 's/(..)\s/\0\t\&/g' | sed 's/$/\t\\\\/' | head -n-1 | tail -n +2
cal | sed '1d;2{h;s/./ /g;x};/^\s*$/b;G;s/\n/ /;s/^...\(.\{15\}\).*/\1/;s/.../ &\t\&/g;s/\&$/\\\\/'
b=`echo "$a" | sed 's/./\L&/g'`
b=`echo "$a" | sed 's/./\U&/g'`
sed 's/.*/\L&/'
readlink -f /x/y/../../a/b/z/../c/d
ln -sf "$(readlink -f "$link")" "$link"
od | cut -b 8- | xargs -n 1 | sort | uniq | wc -l
CLEAN=`echo -n $CLEAN | tr A-Z a-z`
var1=`echo $var1 | tr '[A-Z]' '[a-z]'`
head /dev/random -c16 | od -tx1 -w16 | head -n1 | cut -d' ' -f2- | tr -d ' '
find $(pwd) -type f | xargs -I xxx sed -i 's/\r//g' xxx
cp -f "$project_dir"/iTunesArtwork Payload/iTunesArtwork
cp "${FILE}" "COLLECT/$(mktemp job_XXXXXXXXX)"
cp -v [MacVim_source_folder]/src/MacVim/mvim /usr/local/bin
sudo cp -a libgtest_main.so libgtest.so /usr/lib/
cp -n src dest
find . -name '*FoooBar*' | sed 's/.*/"&"/' | xargs cp ~/foo/bar
find . -name '*FooBar*' -exec cp -t ~/foobar -- {} +
find . -name "*foo*" | sed -e "s/'/\\\'/g" -e 's/"/\\"/g' -e 's/ /\\ /g' | xargs cp /your/dest
find . -type f -name '*.txt' | sed 's/'"'"'/\'"'"'/g' | sed 's/.*/"&"/'  | xargs -I{} cp -v {} ./tmp/
cp lib*.so ~/usr/gtest/lib
find -name '*FooBar*' -print0 | xargs -0 cp -t ~/foo/bar
find . -type f -not -iname '*/not-from-here/*' -exec cp '{}' '/dest/{}' ';'
find . -iname "*foobar*" -exec cp "{}" ~/foo/bar \;
find . -name "file.ext"| grep "FooBar" | xargs -i cp -p "{}" .
find . | grep FooBar | xargs -I{} cp {} ~/foo/bar
cp -n
cp -vi /boot/config-`uname -r` .config
ls | xargs -n 1 cp -i file.dat
ls -d */ | xargs -iA cp file.txt A
echo dir1 dir2 dir3 | xargs -n 1 cp file1
cat allFolders.txt | xargs -n 1 cp fileName.txt
find . -mindepth 1 -maxdepth 1 -type d| grep \/a |xargs -n 1 cp -i index.html
find . -mindepth 1 -maxdepth 1 -type d| xargs -n 1 cp -i index.html
echo ./fs*/* | xargs -n 1 cp test
cp --parents src/prog.js images/icon.jpg /tmp/package
cp $(ls -1tr * | tail -1) /tmp/
rsync --blocking-io *.cc *.h SConstruct rsync://localhost:40001/bledge_ce
rsync -pr ./export /path/to/webroot
rsync --iconv=UTF-8-MAC,UTF-8 /Users/username/path/on/machine/ 'username@server.ip.address.here:/home/username/path/on/server/'
rsync --iconv=UTF-8,UTF-8-MAC /home/username/path/on/server/ 'username@your.ip.address.here:/Users/username/path/on/machine/'
rsync -a --relative /new/x/y/z/ user@remote:/pre_existing/dir/
rsync -r /path/to/source username@computer:/path/to/dest
rsync 6.3.3/6.3.3/macosx/bin/mybinary ~/work/binaries/macosx/6.3.3/
cat allFolders.txt | xargs -n 1 cp fileName.txt
echo 'some_file_name' | cpio -p --owner someuser:somegroup destination_directory
rsync -av --exclude='path1/to/exclude' --exclude='path2/to/exclude' source destination
rsync -u src dest
rsync -R src/prog.js images/icon.jpg /tmp/package
rsync -Rv src/prog.js images/icon.jpg /tmp/package/
rsync -r username@computer:/path/to/source /path/to/dest
find . -type f -name "*.mp3" -exec cp {} /tmp/MusicFiles \;
find dir/ -name '*.txt' | xargs cp -a --target-directory=dir_txt/ --parents
find "$somedir" -type d -empty -exec cp /my/configfile {} \;
find . -depth -print | cpio -o -O /target/directory
find . -name "*c" -print0 | xargs -0 -n1 cp xyz.c
find ./C -name "*.c" | xargs -n1  cp xyz.c
find . -type d -name "temp*" | xargs -n1 cp xyz.c
rsync --sparse sparse-1 sparse-1-copy
find ./ -mount -depth -print | cpio -pdm /destination_dir
find projects/ -name '*.php' -print | cpio -pdm copy/
find . -name \*.xml -print0 | cpio -pamvd0 /new/parent/dir
find /source_path -name *.data -exec cp {} /target_path \;
find . -type f -name "*.mp3" -exec cp {} /tmp/MusicFiles \;
find -name '*.patch' -print0 | xargs -0 -I {} cp {} patches/
find ./work/ -type f -name "*.pdf" -mtime +5 -size +2M  | xargs -r cp -t ./backup/
find ~/ -name *.png -exec cp {} imagesdir \;
find dir/ -name '*.txt' | xargs cp -a --target-directory=dir_txt/ --parents
rsync -a --include='*/' --exclude='*' source/ destination/
rsync -a -f"+ */" -f"- *" source/ destination/
rsync /path/to/local/storage user@remote.host:/path/to/copy
find /home/ -maxdepth 1 -print | sudo cpio -pamVd /newhome
find -print0 | sort -z | cpio -pdv0 ../new
find -name '*FooBar*' -print0 | xargs -0 cp -t ~/foo/bar
find . | grep FooBar | xargs -I{} cp {} ~/foo/bar
find . -iname "*foobar*" -exec cp "{}" ~/foo/bar \;
find folder* -name '*.a' -print | cpio -pvd /path/to/dest
find . | cpio -pdumv /path/to/destination/dir
find /mail -type f | cpio -pvdmB /home/username
find /var/spool/mail -type f | cpio -pvdmB /home/username/mail
find . -type f -not -path '*/exlude-path/*' -exec cp --parents '{}' '/destination/' \;
find . -type f -not -iname '*/not-from-here/*' -exec cp '{}' '/dest/{}' ';'
find . -type f -not -path '*/not-from-here/*' -exec cp '{}' '/dest/{}' \;
cp `ls | grep -v Music` /target_directory
find . -type f | xargs grep -l "textToSearch" | cpio -pV $destination_path
rsync -zvr --include="*.sh" --exclude="*" $from/*  root@$host:/home/tmp/
find . -name "*failed.ipynb" | cpio -pd ./fails
find . -name 'file_name.extension' -print | cpio -pavd /path/to/receiving/folder
find olddir -name script.sh -printf "%p\0" -printf "newdir/%P\0" | xargs -0L2 cp -n
find . | grep "FooBar" | tr \\n \\0 | xargs -0 -I{} cp "{}" ~/foo/bar
find myfiles | cpio -pmud target-dir
find foo -type f ! -name '*Music*' -exec cp {} bar \;
find  /home/mine -iname "*.png" -printf "%P\n " | xargs  -I % -n1 cp %  /home/mine/pngcoppies/copy%
find /home/mine -iname "*.png" -execdir cp {} /home/mine/pngcoppies/copy{} \;
find "/tmp/2/" -iname "$j.sh" -exec cp {} "$i" \;
find . -type f -exec cp -t TARGET {} \+
find /path -type f -name '*~' -print0 | xargs -0 -I % cp -a % ~/backups
find src/ -type d -exec mkdir -p dest/{} \; -o -type f -exec touch dest/{} \;
yes | cp -rf /zzz/zzz/* /xxx/xxx
find dir/ -name '*.txt' | xargs cp -a --target-directory=dir_txt/ --parents
find "$sourceDir" -type d | sed -e "s?$sourceDir?$targetDir?" | xargs mkdir -p
find . -type d -exec mkdir -p -- /path/to/backup/{} \;
find olddir -type d -printf "newdir/%P\0" | xargs -0 mkdir -p
find . -depth -print | cpio -o -O /target/directory
cp /file/that/exists /location/for/new/file
cp -n src dest
find dir1 dir2 dir3 dir4 -type d -exec cp header.shtml {} \;
scp -C file remote:
cp --remove-destination `readlink bar.pdf` bar.pdf
cp --remove-destination `readlink file` file
sudo cp -a include/gtest /usr/include
cp -rf --remove-destination `readlink file` file
cat $1 | ssh $2 "mkdir $3;cat >> $3/$1"
chmod --reference version2/somefile version1/somefile
rsync -rtvpl /source/backup /destination
rsync -avz --chmod=o-rwx -p tata/ tata2/
cp -nr src_dir dest_dir
cp --parents src/prog.js images/icon.jpg /tmp/package
find ./ -depth -print | cpio -pvd newdirpathname
find . | cpio -pdumv /path/to/destination/dir
find original -type d -exec mkdir new/{} \;
find . -type d | cpio -pdvm destdir
find src/ -type d -exec mkdir -p dest/{} \; -o -type f -exec touch dest/{} \;
rsync -rl --delete-after --safe-links pi@192.168.1.PI:/{lib,usr} $HOME/raspberrypi/rootfs
find . | cpio -pdumv /path/to/destination/dir
cp -R t1/ t2
cp `which python2.7` myenv/bin/python
chown --reference=file.txt -- "$tempfile"
chown --reference=oldfile newfile
bash | tee /var/log/bash.out.log
find /your/webdir/ -type d -print0 | xargs -0 chmod 755
find /your/webdir -type f | xargs chmod 644
find . -maxdepth 1 -type d -exec ls -dlrt {} \; | wc --lines
find . -type d -exec ls -dlrt {} \; | wc --lines
find /DIR -type f -print0 | tr -dc '\0' | wc -c
find . -name "*.c" -print0 | xargs -0 cat | wc -l
find -name '*php' | xargs cat | wc -l
find -name '*.php' | xargs cat | wc -l
find . -name "*.php" | xargs grep -v -c '^$' | awk 'BEGIN {FS=":"} { $cnt = $cnt + $2} END {print $cnt}'
find ~music -type f -iname *.mp3 | wc -l
find . -name '*.php' | xargs wc -l
find . -atime +30 -exec ls \; | wc -l
find "$DIR_TO_CLEAN" -mtime +$DAYS_TO_SAVE | wc -l
find . -maxdepth 1 -type f -printf '%TY-%Tm\n' | sort | uniq -c
find /home/my_dir -name '*.txt' | xargs grep -c ^.*
cat foo.pl | sed '/^\s*#/d;/^\s*$/d' | wc -l
find /path/to/dir/ -type f -name "*.py" -exec md5sum {} + | awk '{print $1}' | sort | md5sum
cat foo.c | sed '/^\s*$/d' | wc -l
sed '/^\s*$/d' foo.c | wc -l
result="$(dig +short @"$server" "$domain" | wc -l)"
zcat Sample_51770BL1_R1.fastq.gz | wc -l
zcat *R1*.fastq.gz | wc -l
echo "123 123 123" | grep -o 123 | wc -l
who | awk -F' ' '{print $1}' | sort -u | wc -l
find /usr/src -name "*.html" -exec grep -l foo '{}' ';' | wc -l
find /usr/src -name "*.html" | xargs grep -l foo | wc -l
sed 's/[^x]//g' filename | tr -d '\012' | wc -c
find /home/user1/data1/2012/mainDir -name '*.gz' | wc -l
find -name "*.gz" | wc -l
find . -name "*.java" | wc -l
find . -mindepth 1 -maxdepth 1 -type d | wc -l
find /mount/point -maxdepth 1 -type d | wc -l
diff -U 0 file1 file2 | grep ^@ | wc -l
find . -type f | xargs | wc -c
diff file1 file2 | grep ^[\>\<] | wc -l
diff -U 0 file1 file2 | grep -v ^@ | wc -l
find . -type f -exec basename {} \; | wc -l
find /directory/ -maxdepth 1 -type d -print| wc -l
comm -12 <(sort file1.txt) <(sort file2.txt) | wc -l
comm -12 ignore.txt input.txt | wc -l
find /usr/ports/ -name pkg-plist\* -exec grep dirrmtry '{}' '+' | wc -l
find /usr/ports/ -name pkg-plist\* -exec grep -l etc/rc.d/ '{}' '+' | wc -l
find /usr/ports/ -name pkg-plist\* -exec grep 'unexec.rmdir %D' '{}' '+' | wc -l
find . -type d -exec basename {} \; | wc –l
find /dev/sd*[a-z] -printf . | wc -c
find /dev/sd*[a-z] | wc -l
find /data/SpoolIn -name job.history -exec grep -l FAIL {} \+ | wc -l
find /data/SpoolIn -name job.history -exec grep -l FAIL {} \; | wc -l
find /data/SpoolIn -name job.history | xargs grep -l FAIL | wc -l
find -name file1 | wc -l
find -name file1 | wc -l
find . -name "*.php" -exec chmod 755 {} \; -exec /bin/echo {} \; | wc -l
cat /dir/file.txt | wc -l
cat /etc/fstab | wc -l
cat myfile.txt | wc -l
fold -w "$COLUMNS" testfile | wc -l
find . -name '*.php' -type f | xargs cat | wc -l
wc -l `tree -if --noreport | grep -e'\.php$'`
cat *.txt | wc -l
find xargstest/ -name 'file??' | sort | xargs wc -l
find . -name "*.java" -exec wc -l {} \;
find . -name "*.rb" -type f -exec wc -l \{\} \;
find . -name "*.rb" -type f -print0 | xargs -0 wc -l
who | grep -v localhost | wc -l
watch "ls /proc/$PYTHONPID/fd | wc -l"
find ${DIRECTORY} -type f -print | sed -e 's@^.*/@@' | grep '[aeiouyAEIOUY]' | wc -l
find . -maxdepth 1 -type f -iname '*[aeiouy]*' -printf ".\n" | wc -l
find . -type f | wc -l
find . -type f -perm 755 | wc -l
find teste2 -type f -iname "$srchfor"|wc -l
find ./randfiles/ -type f | wc -l
who | awk '{print $1}' | sort | uniq -c | sort -n
zcat file.gz | awk -v RS="-----------\n" '/A=2[ ,\n]/ && /dummy=2[ ,\n]/{count++} !/dummy=2[ ,\n]/{other++} END{print "Final counter value=",count, "; other=", other}'
zcat file.gz | awk -v RS="-----------\n" '/A=2[ ,\n]/ && /dummy=2[ ,\n]/{count++} END{print "Final counter value=",count}'
find . -print0 | tr -cd '\0' | wc -c
find . -type f -name "*.*" | grep -o -E "\.[^\.]+$" | grep -o -E "[[:alpha:]]{3,6}" | awk '{print tolower($0)}' | sort | uniq -c | sort -rn
sort file1 file2 | uniq -d | wc -l
find . -type f | sed -e 's/.*\.//' | sed -e 's/.*\///' | sort | uniq -c | sort -rn
comm -23 a.txt b.txt | wc -l
who | wc -l
who | sed 1d | wc -l
echo "1 1 2 2 2 5" | tr ' ' $'\n' | grep -c 2
find . -name *.py -exec wc -l {} \; | awk '{ SUM += $0} END { print SUM }'
find . -type f -name '*.gz' | xargs zcat | wc -l
find /usr/src -name "*.html" -execdir /usr/bin/grep -H "foo" {} ';' | wc -l
find .  -type f  -name '*.txt' -exec wc -l {} \; | awk '{total += $1} END{print total}'
wc -l `find . -type f -name '*.txt' `
find . -type f -exec wc -l {} \; | awk '{ SUM += $0} END { print SUM }'
ls -l /boot/grub/*.mod | wc -l
cal -h | cut -c 4-17 | tail -n +3  | wc -w
find DIR_NAME -type f | wc -l
find -type f -printf '\n' | wc -l
find . -type f | wc -l
find `pwd` -type f -exec ls -l {} \; | wc -l
cat $i | wc -l
sed '/^\s*$/d' $i | wc -l ## skip blank lines
ls -1 | wc -l
wc -l `find . -type f \( -name "*.cpp" -o -name "*.c" -o -name "*.h" \) -print`
find . -name '*.php' -type f | sort | xargs wc -l
find . -name '*.php' -type f | xargs wc -l | sort -nr
find . -name '*.php' -type f | xargs wc -l
wc -l $(find . -name "*.php")
wc -l `find . -name "*.php"`
wc -l `tree -if --noreport | grep -e'\.php$'`
find . -name "*.php" | xargs wc -l
wc -l **/*.php
find . -name '*.php' | awk '{gsub(" ","\\ ", $0);print $0}' |xargs wc -l
find . -name tests -prune -o -type f -name '*.php' | xargs wc -l
find . -name "*.php" -not -path "./tests*" | xargs wc -l
find . -name '*.php' | xargs wc -l | sort -r
wc -l $file | awk '{print $1}';
cat $file | sed '/\/\//d' | sed '/^\s*$/d' | wc -l
cat 'filename' | grep '[^ ]' | wc -l
awk '!/^[[:space:]]*$/{++x} END{print x}' filename
wc -l file
cat /dir/file.txt | wc -l
wc -l /dir/file.txt
cat /etc/fstab | wc -l
cat *.txt | wc -l
cat myfile.txt | wc -l
grep -n -i null myfile.txt | wc -l
grep -v '^\s*$' *.py | wc
cat fileName | grep -v ^$ | wc -l
find . -name '*.php' | wc -l
cat ip_addresses | sort | uniq -c | sort -nr | awk '{print $2 " " $1}'
NUMCPU=$(grep $'^processor\t*:' /proc/cpuinfo |wc -l)
find . -name "*.php" | xargs grep -v -c '^$' | awk 'BEGIN {FS=":"} { cnt = cnt + $2} END {print cnt}'
find . -name '*.php' -o -name '*.inc' | xargs wc -l
find . -name '*.js' -or -name '*.php' | xargs wc -l | grep 'total'  | awk '{ SUM += $1; print $1} END { print "Total text lines in PHP and JS",SUM }'
find . -name '*.php' -type f | xargs cat | wc -l
find . -name '*.php' | xargs cat | awk '/[a-zA-Z0-9]/ {i++} END{print i}'
find . -type f -name "*" -newermt 2013-12-04 ! -newermt 2013-12-05 | xargs -I {} tar -czvf files.tar.gz {}
seq 1 1000 | split -l 1 -a 3 -d - file
echo "a.txt b.txt c.txt d.txt z.txt" | xargs touch
tmp=$(mktemp -d $(dirname "$1")/XXXXXX)
tmpfile=$(mktemp $(dirname "$file1")/XXXXXX)
tmpfile=$(mktemp $(dirname "$1")/XXXXXX)
mkdir -p folder$( seq -s "/folder" 999 )1000
tar czf - www|split -b 1073741824 - www-backup.tar.
tar -czvf my_directory.tar.gz -C my_directory .
tar --one-file-system -czv /home | split -b 4000m - /media/DRIVENAME/BACKUPNAME.tgz
find . -mindepth 1 -maxdepth 1 -type d| grep \/a |xargs -n 1 cp -i index.html
find . -mindepth 1 -maxdepth 1 -type d| xargs -n 1 cp -i index.html
find . -type d -print|sed 's@^@/usr/project/@'|xargs mkdir
find . -type d -print|sed 's@^@/usr/project/@'|xargs mkdir -p
mkdir alpha_real
ln -s $(readlink -f $origlink) $newlink
find $sourcePath -type f -name "*.log" -exec tar -uvf $tarFile {} \;
ln myfile.txt my-hard-link
ssh -i keyfile.rsa -T -N -L 16379:localhost:6379 someuser@somehost
echo -e  'y\n'|ssh-keygen -q -t rsa -N "" -f ~/.ssh/id_rsa
rsync /dev/null node:existing-dir/new-dir/
find /path/to/drive -type f -exec file -b '{}' \; -printf '%s\n' | awk -F , 'NR%2 {i=$1} NR%2==0 {a[i]+=$1} END {for (i in a) printf("%12u %s\n",a[i],i)}' | sort -nr
ssh-keygen -b 2048 -t rsa -f key -C michael
ssh-keygen -t rsa -C "$APP" -N "$SSHKEYPASS" -f ~/.ssh/id_rsa
ssh-keygen -f ~/.ssh/apache-rsync
ssh-keygen -t rsa
ssh-keygen -f outfile -N ''
ssh -N -L 2222:remote.example.com:22 bridge.example.com&
ln -s target
ln -s target-directory/`ls -rt target-directory | tail -n1` latest
ln -s `pwd`/current/app/webroot public_html
find /foo -maxdepth 1 -type f ! -name 'runscript*' -exec ln -s {} /bar/tmp/ \;
find $PWD -name '.[^.]*' -exec ln -s '{}' /path/to/dir \;
find original -type f -exec ln -s {} new/{} \;
cat results2.txt | xargs -I{} ln -s {} ~/newlinks
find $PWD -type f -exec ln -st $tmpdir {} +
ln -s "$source" -t ~/newlinks
ln -s "$(find dir -name '*.jpg')" .
find /home/michael/foxpro/mount/A[1FV]/[12][0-9][0-9][0-9] -name '*.dbf' -type f -exec ln -s {} \;
find ../[12][0-9][0-9][0-9] -type f -exec ln -s {} \;
ln -s git-stuff/home/.[!.]* .
ln -s "$file"
ln -s "../config/init"
ln -s "`pwd`" $1/link
ln -s "$(readlink -e "$2")" "$1/link"
ln -s $HOME/downloads/fnord $HOME/bin/
ln -s "$ACTUAL_DIR" "$SYMLINK"
ln -s "$(which bam2)" "$tmpdir"/bam
ln -s  "${TARGET}${file}"  "${DESTINATION}${file}"
ln -s .bashrc .bash_profile
ln -sn git-stuff/home/profile .profile
ln -s /lib/libc.so.6 /lib/libc.so.0
ln -sF /usr/share/my-editor/my-editor-executable   /usr/bin/my-editor
sudo ln -s "/Applications/Sublime Text 2.app/Contents/SharedSupport/bin/subl" /usr/local/bin/subl
ln -s   /var/cache/apt/archives/bash_4.3-14ubuntu1_amd64.deb foo
ln -s newtarget temp
ln -s "$wh" wh
ln -s "/Applications/Sublime Text 2.app/Contents/SharedSupport/bin/subl" ~/bin/subl
ln -s "/Applications/Sublime Text.app/Contents/SharedSupport/bin/subl" ~/bin/subl
ln -s $file `basename $file`
ln -r -s "$orig_dest" "$dest_dir/$orig_name"
ln $file /tmp/allfiles
sudo ln -s /usr/bin/perl /usr/local/bin/perl`echo -e '\r'`
ln -s "$dir" "$1/link"
ln -s /Applications/Sublime\ Text.app/Contents/SharedSupport/bin/subl /usr/local/
ln -s /Applications/Sublime\ Text.app/Contents/SharedSupport/bin/subl /usr/local/bin/
ln -s /Applications/Sublime\ Text\ 2.app/Contents/SharedSupport/bin/subl /usr/local/bin/
join -a1 -a2 <(sed s/^Gene/00ne/ S43.txt | sort) <(sed s/^Gene/00ne/ S44.txt | sort) | column -t | sed s/^00ne/Gene/
echo -en '111 22 3\n4 555 66\n' | column -t | sed 's/ \([0-9]\)/\1/g'
find . -type f -name "*.java" | xargs tar cvf myfile.tar
find ~/Library -name '* *' -print0 | xargs -0 tar rf blah.tar
source  <(date +"CDATE='%Y-%m-%d %H:%M:%S' EPOCH='%s'")
sudo mkdir -p $javaUsrLib
mkdir -p $tempWork
find ./test -printf "././%f\n"| cpio -o -F newArch
find -name "bar" -execdir touch foo \;
touch -m 201111301200.00 $log_dir/last.check
touch "$correctFilePathAndName"
cp /dev/null emptyfile.c
find . -type d -name "mydir" -exec touch '{}/abc.txt' \;
find . -type d -name "mydir" -print |  sed 's/$/\/abc.txt/g' | xargs touch
echo -e "Icon\\r" | xargs touch
touch $'Icon\r'
find . -type d -exec touch {}/index.html \;
touch index.html
find . -mindepth 1 -maxdepth 1 -type d | awk 'BEGIN {FS="./"}; {print $2}' | xargs -d '\n' tar czf backup1.tar
tar cz my_large_file_1 my_large_file_2 | split -b 1024MiB - myfiles_split.tgz_
find . -type f -mtime -7 -print -exec cat {} \; | tar cf - | gzip -9
mkdir -p ./some/path
rsync -a --rsync-path="mkdir -p /tmp/x/y/z/ && rsync" $source user@remote:/tmp/x/y/z/
mkdir dir2
echo -e "$correctFilePathAndName" | xargs touch
find test -path 'test/icecream/cupcake/*' -o -path 'test/mtndew/livewire/*' | cpio -padluv test-keep
mkdir -p a/b/c
mkdir -p `dirname /full/path/to/file.txt`
mkdir -p tmp/real_dir1 tmp/real_dir2
mkdir -p x/p/q
mkdir -p $2
mkdir -p /my/other/path/here
mkdir -p /tmp/test/blah/oops/something
mkdir -p project/{lib/ext,bin,src,doc/{html,info,pdf},demo/stat/a}
mkdir -p directory{1..3}/subdirectory{1..3}/subsubdirectory{1..2}
mkdir -p foo/bar/baz
mkdir -p ~/foo/bar/baz ~/foo/bar/bif ~/foo/boo/bang
mkdir -p path2/{a..z}
find . -type f -mtime +1000 -print0 | cpio -dumpl0 /home/user/archives
ssh -M -S my-ctrl-socket -fnNT -L 50000:localhost:3306 jm@sampledomain.com
cat <(fgrep -i -v "$command" <(crontab -u test -l)) <(echo "$job") | crontab -u test -
cat <(fgrep -i -v "$command" <(crontab -l)) <(echo "$job") | crontab -
ssh -L 4444:raptor.lan:22 genja.org
ln -s "$(readlink -e "$2")" "$1/link"
find dir -name '*.jpg' -exec ln -s "{}" \;
find /your/project -maxdepth 1 ! -name "CONFIGFILE" -exec ln -s \{\} ./ \;
find /your/project -type f ! -name 'CONFIGFILE' -exec ln -s \{\} ./ \;
find /path/with/files -type f -name "*txt*" -exec ln -s {} . ';'
find bar1 -name '*foo*' -not -type d -not -name '*.cc' -exec ln -s $PWD/'{}' bar2/ \;
find /home/folder1/*.txt -type f -exec ln -s {} "folder1_" +\;
find /home/folder1/*.txt -type f -exec ln -s {} "folder2_" + \;
find /home/folder1/*.txt -type f -exec ln -s {} \;
find /home/folder2/*.txt -type f -exec ln -s {} \;
find /tmp/a1 -exec tar -rvf dirall.tar {} \;
find /tmp/a1 | xargs tar cvf foo.tar
find /home/testuser/log/ -mtime +1 | xargs  tar -czvPf  /opt/older_log_$(date +%F).tar.gz
mkdir ~/.npm-global
mkdir "${HOME}/.npm-packages"
rand_str="$(mktemp --dry-run XXXXX)"
ifconfig eth0:fakenfs 192.0.2.55 netmask 255.255.255.255
dir="$(mktemp aws-sync-XXXXX)"
mkdir -p -- "$(dirname -- "$f")"
set script_dir = `pwd`/`dirname $0`
rand_str=$(mktemp --dry-run ${str// /X})
my_tmp_dir=$(mktemp -d --tmpdir=/tmp)
TMPDIR=$(mktemp -p /tmp -d .daemonXXXXXXX)
content_dir1=$(mktemp)
content_dir2=$(mktemp)
fif2=$(mktemp -u)
tmpfile=$(mktemp)
tmpfile=`mktemp`
mktemp -t identifier.XXXXXXXXXX
launcherfile=$(mktemp -p "$appdir" "$template")
TMPPS_PREFIX=$(mktemp "${TMPDIR:-/tmp/}${tempname}.XXXXXX")
tempFile="$(mktemp "${TMPDIR:-/tmp/}$(basename "$0")-XXXXX")"
mktemp
f=`mktemp -p .`
tempfile=$(mktemp $(pwd)/templateXXXXXX)
source=`mktemp`
TMP_FILE="$(mktemp -t)"
LGT_TEMP_FILE="$(mktemp --suffix .cmd)"
gnuplotscript=$(mktemp /tmp/gnuplot_cmd_$(basename "${0}").XXXXXX.gnuplot)
script1=`mktemp /tmp/.script.XXXXXX`;
script2=`mktemp /tmp/.script.XXXXXX`;
tmp_file=`mktemp --tmpdir=/tmp emacs-manager.XXXXXX`
mytemp="$(mktemp -t "${PROG}")"
mktemp /tmp/banana.XXXXXXXXXXXXXXXXXXXXXXX.mp3
TMPDIR=$(mktemp -d)
other="$(mktemp --directory)"
td=$( mktemp -d )
tempd=`mktemp -d`
mktemp -d -t
dir=$(mktemp -d)
tmpdir=$(mktemp -d)
$ my_temp_dir=$(mktemp -d --tmpdir=$temp_dir -t $template)
mydir=$(mktemp -d "${TMPDIR:-/tmp/}$(basename $0).XXXXXXXXXXXX")
tmpdir=$(mktemp -d /tmp/tardir-XXXXXX)
mktemp -d -p /path/to/dir
mktemp -dt "$(basename $0).XXXXXXXXXX"
rsync_src=`mktemp -d -p $mnt_dir`
tdir="$(pwd)/$(mktemp -d)"
sed -n 's;\(http://[^/]*\)/.*;\1;p'
sed -n 's;\(http://[^/]*/\).*;\1;p'
echo '1234567890  *' | rev | cut -c 4- | rev
rmdir "$(dirname $dir)"
rmdir "$(dirname $(dirname $dir))"
gzip -dc /file/address/file.tar.gz
awk -F'\t' 'NR==FNR{a[$5];next} $5 in a' <(zcat file2.txt) <(zcat file1.txt)
gzip -dc path/to/test/file.gz | grep -P 'my regex' | grep -vP 'other regex' | split -dl1000000 - file
gzip -dc path/to/test/file.gz | grep -P --regexp='my regex' | split -dl1000000 - file
gzip -dc path/to/test/file.gz | grep -P --regexp='my regex' | split -l1000000
gzip -d ${set1[@]} &
gzip -d file.gz
gzip -d --stdout file.gz | bash
gzip -dc /usr/src/redhat/SOURCES/source-one.tar.gz | tar -xvvf -
gzip -dc archive.tar.gz | tar -xf - -C /destination
gzip -dc libxml2-sources-2.7.7.tar.gz | tar xvf -
sort -m <(zcat $part0 | sort) <(zcat $part1 | sort)
gzip -dc hello-0.2.tar.gz | tar -xf -
find . -name "*.bz2" -print0 | xargs -I{} -0 bzip2 -dc {} | cut -f, -d4
bzip2 -dc xac.bz2
bzip2 -d /tmp/itunes20140618.tbz
find test -name ".DS_Store" -delete
COLUMN=`echo $1 | tr -d -`
find /var/www -maxdepth 4 -name 'restore.php' -exec rm -r {} \;
find . -name "*.zip" -mtime +2 -print0 | xargs -0 -I {} rm {}
find . -name "*.zip" -mtime +2 orint0 | xargs -0 rm
find . -name "*txt" -type f -print | xargs rm
find . -name "*.bam" | xargs rm
find . -name "*.pyc" | xargs -0 rm -rf
find . -name "*.pyc" | xargs rm -rf
find . -type d -name .svn -print0|xargs -0 rm -rf
find . -depth -name .svn -exec rm -fr {} \;
find . -name .svn -delete
find . -name .svn -exec rm -rf '{}' \;
find . -name .svn -exec rm -rf {} +
find . -name .svn -exec rm -rf {} \;
find . -name .svn -exec rm -v {} \;
find . -name .svn | xargs rm -fr
find . -name .svn |xargs rm -rf
rm -rf `find . -type d -name ".svn"`
find . -iname "1US*" -exec rm {} \;
find ~/mydir -iname '*.htm' -exec rm {} \;
find /tmp -iname '*.mp3' -print0 | xargs -0 rm
find . -maxdepth 1 -type d -name '__temp__*' -print0 | xargs -0 rm -rf
find . -depth -name '__temp__*' -exec rm -rf '{}' \;
find . -name __temp__* -exec rm -rf '{}' \;
find -L /usr/ports/packages -type l -exec rm -- {}	+
ls -tr | head -n -5 | xargs rm
find / -maxdepth 1 -xdev -type f -exec grep -i "stringtofind" -l {} \; -exec sed -i '/./d' {} \;
find / -maxdepth 1 -xdev -type f -exec grep -i "stringtofind" -q "{}" \; -print0 | xargs -0 sed '/./d'
find $LOGDIR -type d -mtime +5 -exec rm -rf {} \;
find /myDir -type d -delete
find /TBD -mtime +1 -type d | xargs rm -f -r
find .cache/chromium/Default/Cache/ -mindepth 1 -type d -size +100M -delete
find .cache/chromium/Default/Cache/ -mindepth 1 -type d -size +100M -exec rm -rf {} \;
find -type d -empty -exec rmdir -vp --ignore-fail-on-non-empty {} +
find directory -mindepth 1 -type d -empty -delete
find root -mindepth 2 -type d -empty -delete
find test -depth -type d -empty -delete
find -type d -empty -exec rmdir -vp --ignore-fail-on-non-empty {} +
find -type d -empty
find root -type -d -empty -delete
find test -depth -type d -empty -delete
find test -depth -empty -delete
find . -type f -empty -delete
find test -depth -empty -delete
find directory -mindepth 1 -type d -empty -delete
find /dir -name "filename*" -type f -delete
find /dir -name "filename*" -type f -exec rm {} \;
find /dir -name "filename*" -type f -print | xargs rm
find . -name "core" -exec rm -f {} \;
find -delete
find . -type f -name "Foo*" -exec rm {} \;
find "$DIR" -type f -atime +5 -exec rm {} \;
find "${S}/bundled-libs" \! -name 'libbass.so' -delete
find "$DIR" -type f -atime +5 -exec rm {} \;
find /TBD/* -mtime +1 -exec rm -rf {} \;
find /myDir -mindepth 1 -mtime 7 -delete
find /myDir -mindepth 1 -mtime 7 -exec rm -rf {} \;
find /myDir -mindepth 1 -mtime +7 -delete
find /myDir -mtime 7 -exec rm -rf {} \;
find /TBD/* -mtime +1 | xargs rm -rf
find . -name "*~" -delete
find . -name "*~" -exec rm {} \;
find . -exec /bin/rm {} \;
find ~ -atime +100 -delete
find . -name "filename" -and -not -path "./path/to/filename" -delete
find . -name "filename" -and -not -path "*/myfolder/filename" -delete
find . -name "-F" -exec rm {} \;
find ~/Books -type f -name Waldo -exec rm {} \;
sudo find /home/user/Series/ -iname sample -exec rm {} \;
find / -nouser | xargs -0 rm
find . -type f -atime +30 -exec rm {} \;
find /path-to-directory -mtime +60 -exec rm -f {} \;
find / -nouser | xargs -0 rm
find $DESTINATION -mtime +7 -exec rm {} \;
find $INTRANETDESTINATION/monthly -mtime +366 -exec rm {} \;
find $INTRANETDESTINATION/weekly -mtime +32 -exec rm {} \;
find /home/backups -type f -iregex '.*\.t?gz$' -mtime +60 -exec rm {} \;
find /path/to/files -type f ! -newer dummyfile -delete
find /path/to/input/ -type f -exec grep -qiF spammer@spammy.com \{\} \; -delete
find -mindepth 1 -delete
find . -name "*.$1" -exec rm {} \;
find root -type -f -cmin +30 -delete
find ~ -used +365 -ok rm '{}' ';'
find project / src / -name "* .o" -exec rm -f {} \;
find . -name “*.old” -exec rm {} \;
find . -name “*.old” -print | xargs rm
find . -inum 128128 | xargs rm
find . -inum $inum -exec rm {} \;
find -inum 804180 -exec rm {} \
find root -mindepth 2 -delete
find . -name "FILE-TO-FIND" -exec rm -rf {} \;
find -name file -delete
sudo find /home/user/Series/ -iname sample -print0 | sudo xargs -0 rm -r
find . -maxdepth 2 -name "test" -exec rm -rf {} \;
find /path/to/files* -mtime +2 -delete
find * -maxdepth 0 -name 'b' -prune -o -exec rm -rf '{}' ';'
find * -maxdepth 0 -name 'b' -prune -o -exec rm -rf {} \;
find -delete
find . ! -name '.gitignore' ! -path '.git' ! -path '.git/*' -exec rm -rf {} \;
find . -name "*.$1" -delete;
find . -name “*.old” -delete
find -inum 117672808 -exec rm {} \;
find . -inum $inum -exec rm {} \
find /home -xdev -inum 2655341 | xargs rm
find $FOLDER -name ".*" -delete
find "$some_directory" -type f -name '.*' -delete
find "$some_directory" -type f -name '.*' -exec rm '{}' \;
find "$some_directory" -type f -name '.*' | xargs rm
find "$some_directory" -name '.*' ! -name '.' ! -name '..' -delete
find $some_directory '.*' -delete
find . -delete
sed -i.bak '/pattern to match/d' ./infile
sed -i '/pattern/d' filename
sed --in-place '/some string here/d' yourfile
find ./ -type f -name \*.php -exec sed -i ’s/^.*iframe bla bla bla.*$//g’ {} \;
echo "${depsAlastmodified[$i]}" | tr -cd '[[:digit:]]' | od -c
find . -type f -name "FILE-TO-FIND" -exec rm -f {} \;
find . -type f -name "FindCommandExamples.txt" -exec rm -f {} \;
find . -type f -name "IMAG1806.jpg" -exec rm -f {} \;
find . -type f -name 'IMAGE1806.jpg' -delete
find /path/to/junk/files -type f -mtime +31 -exec rm -f {} \;
find $DIR -type f -mtime +60w -exec rm {} \;
find $OUTPUTDIR -type f -mtime +7 -delete
find -O3 "$save_path" -depth -mindepth 1 -name 'sess_*' -ignore_readdir_race -type f -cmin "+$gc_maxlifetime" -delete
find "$DIR" -type f \! -newer "$a" -exec rm {} +
find . -type f -print0 | xargs -0 /bin/rm
find $HOME/Library/Safari/Icons -type f -atime +30 -name "*.cache" -print -delete
find /home/u20806/public_html -daystart -maxdepth 1 -mmin +25 -type f -name "*.txt" \ -exec rm -f {} \;
find /home/u20806/public_html -maxdepth 1 -mmin +25 -type f -name "*.txt" -delete
find . -type f -inum 314167125 -delete
find . -name "*.c" | xargs rm -f
find . -name heapdump* -exec rm {} \ ;
find . -name heapdump*|xargs rm
find "$DIR_TO_CLEAN" -type -f -mtime "+$DAYS_TO_SAVE" -exec rm {} \; -printf '.' | wc -c
find "$DIR_TO_CLEAN" -type f -mtime +$DAYS_TO_SAVE -print0 | awk -v RS='\0' -v ORS='\0' '{ print } END { print NR }' | xargs -0 rm
finger |  cut --complement -c36-40
finger |  sed 's/\(.\{35\}\)...../\1/'
crontab yourFile.text
find . -type d -empty -delete
find . -empty -delete -print
sed -n "s/^$//;t;p;"
find . -type f -empty -delete
seq 10 | sed '0~2d'
find -mindepth 1 -maxdepth 1 -print0 | xargs -0 rm -rf
find . -type f -inum 314167125 -delete
find /mnt/zip -name "*doc copy" -execdir rm "{}" \;
find . -name "* *" -exec rm -f {} \;
find . -name '*[+{;"\\=?~()<>&*|$ ]*' -maxdepth 0 -exec rm -f '{}' \;
find "$DIR_TO_CLEAN" -mtime +$DAYS_TO_SAVE -exec rm {} \;
find /var/tmp/stuff -mtime +90 -delete
find /var/tmp/stuff -mtime +90 -exec /bin/rm {} \+
find /var/tmp/stuff -mtime +90 -exec /bin/rm {} \;
find /var/tmp/stuff -mtime +90 -execdir /bin/rm {} \+
find /var/tmp/stuff -mtime +90 -print | xargs /bin/rm
find /var/tmp/stuff -mtime +90 -print0 | xargs -0 /bin/rm
find DIR \( -name 2015\* -a \( -name \*album\* -o -name \*picture\* \) \) -delete
find ./ -mtime +31 -delete
find $LOCATION -name $REQUIRED_FILES -type f -mtime +1 -delete
find $LOCATION -name $REQUIRED_FILES -type f -mmin +360 -delete
find . -inum 782263 -exec rm -i {} \;
find . -inum [inode-number] -exec rm -i {} \;
history -d "$1"
find /var/tmp/stuff1 -mtime +90 -delete &
find . -inum 782263 -exec rm -i {} \;
grep -v '^2 ' file | cut -d' ' -f2- | nl -w1 -s' '
find . -name bad -empty -delete
finger | awk -F"\t" -v 'OFS=\t' '{ $4=""; print $0}' | sed 's/\t\{2,\}/\t/'
xargs -n 1 -I '{}' find "$(pwd)" -type f -inum '{}' -delete
sed '/start/,+4d'
ls -tr $(find /home/backups -name '*.gz' -o -name '*.tgz')|head -1|xargs rm -f
find index.html | xargs -rt sed -i 's/<script>if(window.*<\/script>//g'
rmdir nonsense_dir
find /tmp/*/* -mtime +7 -type d -exec rmdir {} \;
column -s: -t
find /etc -execdir echo "{}" ';'
who am i | awk '{print $5}' | sed 's/[()]//g' | cut -f1 -d "." | sed 's/-/./g'
df /full/path | grep -q /full/path
who | grep $USER
who -m
set +e
set +m
set -f
set -o noclobber
shopt -u compat31
shopt -u dotglob
shopt -u nocasematch
shopt -u nullglob
echo $line | cut -c2- | md5sum
column -x -c 30 /tmp/file
cat infile | od -c
echo 12345 | rev
echo 798|rev
od -t x1 file|cut -c8-
od -c oldfile
find -P . -type f | rev | cut -d/ -f2- | rev | cut -d/ -f1-2 | cut -d/ -f2- | sort | uniq -c
ls -l | more
od -t fD file
od -t fD
find $dir -type -f size +$size -print0 | xargs -0 ls -1hsS
find /your/dir -type f -size +5M -print0 | xargs -0 ls -1Ssh
find $STORAGEFOLDER -name .todo  -exec ls -l {} \;
find . -perm 0777 -type d -exec ls -l {} \;
find /nas -type d -ls
find -name file -ls
find /var/ -size +10M -ls
find /usr/bin -type f -size -50c -exec ls -l '{}' ';'
find . -perm 0777 -type f -exec ls -l {} \;
find . -newer Nov -ls
find / -type f -user root -perm -4000 -exec ls -l {} \;
find / -type f -user bluher -exec ls -ls {}  \;
find . -type f -newermt ‘Apr 18 23:59:59 EDT 2013’ ! -newermt ‘Apr 20 00:00:00 EDT 2013’ -exec ls -l ‘{}’ \;
find /var/ -size +10M -exec ls -lh {} \;
find /home/backups -printf "%T@ %p\n" | sort -n | head -1 | cut -d" " -f2- | xargs ls -al
od -a test.sh
fold -w1 filename | sort | uniq -c | sort -nr
set | more
find /usr/X11/man/man5 -print
find man5 -print
tree -P "*foo"
find /home -group test
cat dax-weekly.csv | awk '{a[i++]=$0} END {for (j=i-1; j>=0;) print a[j--] }'
cat /boot/System.map-`uname -r` | grep funcname
grep something file | more
find . -type f |sed '/.\/dir[12]\/[^/]*$/d'
find -P .  -maxdepth 1 -type l -exec echo -n "{} -> " \; -exec readlink {} \;
find -P . -type l -exec echo -n "{} -> " \; -exec readlink {} \;
find /home/bozo/projects -mtime -1
set derby
ab=`ps -ef | grep -v grep | grep -wc processname`
yes | cat | more
yes | more
set | grep -A999 '^foobar ()' | grep -m1 -B999 '^}'
set | sed -n '/^foobar ()/,/^}/p'
echo `uname -a | awk '{print $2}'`
uname -srvm
cp --help
diff -y one.txt two.txt
diff /destination/dir/1 /destination/dir/2 -r -x *.xml
diff /destination/dir/1 /destination/dir/2 -r -X exclude.pats
diff -x '*.foo' -x '*.bar' -x '*.baz' /destination/dir/1 /destination/dir/2
diff -y /tmp/test1  /tmp/test2
diff /tmp/test1  /tmp/test2
diff -y a b
diff -Naur dir1/ dir2/
diff -r dir1 dir2
diff -r dir1/ dir2/
diff -ENwbur repos1/ repos2/
diff -y file1 file2
diff -a --suppress-common-lines -y a.txt b.txt
diff dir1.txt dir2.txt
diff <(ls /bin) <(ls /usr/bin)
diff /tmp/ksh-{9725,9781}.log | grep ^\<
rev file.txt
set | grep "^_="
file file-name
which file | xargs file
cat -n file.txt | less
less -N file.txt
uname -i
uname -n
file ascii.txt
file utf8.txt
yes 'c=(╱ ╲);printf ${c[RANDOM%2]}'|bash
yes 'printf \\u$[2571+RANDOM%2]'|bash
awk '{printf "%s ", $0} END {printf "\n"}' inputfile
awk '{printf "%s ", $0}' inputfile
awk '{printf "%s|", $0} END {printf "\n"}' inputfile
awk 1 ORS=' ' file
uname -s -r -v
uname -r
tail -1000 file-with-line-too-long.txt | more
ls *.txt | tee /dev/tty txtlist.txt
find /tmp -user me -ls
uname -m
file -ib "$file"
env | grep '^variable='
find . -maxdepth 1 -name '[!.]*' -printf 'Name: %16f Size: %6s\n'
uname -r | cut -d. -f1-2
uname -r | sed 's/\([0-9]\+\.[0-9]\+\)\..*/\1/'
sed 's/$/p/' file_of_line_numbers | sed -nf - source
file -bi myfile.txt
uname -o
tree -p -u -g -f
tree -p -u -g -f -i
df '/some/directory' | awk '{print $1, $6}'
column -s"${tab}" -t
cat | od -b
od -cAn;
du -sh *
uname -a
man bash | less -Ip "\\\'"
find . -type f -exec ls -s {} \; | sort -n -r | head -5
find . -not -empty -type f -exec ls -s {} \; | sort -n  | head -5
find . -type f -exec ls -s {} \; | sort -n  | head -5
find -type f -exec du -Sh {} + | sort -rh | head -n 5
cat YourFile.txt | more
more YourFile.txt
cat `which ~/f`
column -t file | uniq -w12 -c
cat myfile
zcat sample_0001.gz | sed -e 's/lk=1&//g'
cat text
fold -80 your_file | more
more /var/log/syslog
grep -v '^$\|^#\|^\s*\#' filename | grep -v "^[[:space:]]*$" | more
echo "$a" | awk '{print tolower($0)}'
find home/magie/d2 -type f -perm -u+rx | wc -l
find home/magie/d2 -type f -perm +111 | wc -l
tree -I '3rd*'
diff -y -W 150 file1.cf file2.cf
grep -E -m 1 -n 'old' file | sed 's/:.*$//' - | sed 's/$/s\/old\/new\//' - | sed -f - file
tree /
du -sb /data/sflow_log | cut -f1
$ file /bin/bash
find . -print | grep "/${prefix}${ypatt}"
od -t x1 -An /bin/ls | head
od -c foo |head -2
ssh-keygen -l -E md5 -f /etc/ssh/ssh_host_ecdsa_key.pub
ssh-keygen -l -f /etc/ssh/ssh_host_ecdsa_key.pub
echo $foo | rev | cut -c1-3 | rev
echo "$var" | rev | cut -d: -f1 | rev
echo 'maps.google.com' | rev | cut -d'.' -f 1 | rev
rev file.txt | cut -d/ -f1 | rev
basename /usr/local/svn/repos/example
rev file.txt | cut -d ' ' -f1 | rev
file -i filename
mount | grep "^$path" | awk '{print $3}'
echo -e "Test\rTesting\r\nTester\rTested" | awk '{ print $0; }' | od -a
echo -e "line1\r\nline2" | awk '{ print $0; }' | od -a
echo -e "line1\r\nline2" | od -a
find . -type f -exec wc -l {} +
find . -type f -exec echo {} \; | wc -l
find . -type f -print0 | tr -dc '\0' | wc -c
find . -type d -ls | awk '{print $4 - 2, $NF}' | sort -rn | head
ls | column -c 80
du -a --max-depth=1 | sort -n
du -a -h --max-depth=1 | sort -hr
find -name *.undo -print0 | du -hc --files0-from=-
find . -name “*.old” -print | wc -l
find . -regex ".*\.\(flv\|mp4\)" -type f -printf '%T+ %p\n' | sort | head -n 500
du -sb
finger | sed 's/\t/ /' | sed 's/pts\/[0-9]* *[0-9]*//' | awk '{print $2"\t("$1")\t"$3" "$4" "$5}' | sort | uniq
finger | sed 's/^\([^ ]*\) *\([^ ]*\) *pts[^A-Z]*\([^(]*\).*/\2\t(\1)\t\3/'
echo $var | awk '{gsub(/^ +| +$/,"")}1'
mount -l
w
pstree -a
cal -3
cal -3 12 2120
ifconfig -a
pstree -p -s PID
ifconfig
sort | uniq -c
find . -name "*.andnav" | rename -vn "s/\.andnav$/.tile/"
set +e
set +a
rsync -ave ssh '"Louis Theroux"''"'"'"'"''"s LA Stories"'
curl -s 'http://archive.ubuntu.com/ubuntu/pool/universe/s/splint/splint_3.1.2.dfsg1-2.diff.gz' | gunzip -dc | less
find . -type f -iname \*.mov -printf '%h\n' | sort | uniq | xargs -n 1 -d '\n' -I '{}' echo mkdir -vp "/TARGET_FOLDER_ROOT/{}"
od -t x1 -t a /dev/ttySomething
od -tx2 FILENAME
od -t c file
od -xc filename
od -xcb input_file_name | less
od -xcb testscript.sh
echo 'hi' | od -c
echo `echo "Ho ho ho"` | od -c
cp -rs /mnt/usr/lib /usr/
sudo rsync -pgodt /home/ /newhome/
rsync --exclude='B/' --exclude='C/' . anotherhost:/path/to/target/directory
ping -n 1 %ip% | find "TTL"
sudo ln -sf /usr/local/ssl/bin/openssl `which openssl`
set -x
crontab -e
sudo crontab -u wwwrun -e
tac temp.txt | sort -k2,2 -r -u
set -o history -o histexpand
set -H
set -o history
shopt -s dotglob
shopt -s globstar
shopt -s nullglob
shopt -s autocd
shopt -s cdable_vars
shopt -s cmdhist
shopt -s compat31
shopt -s direxpand
shopt -s expand_aliases
shopt -s failglob
shopt -s histappend
shopt -s lastpipe
shopt -s lithist
shopt -s nocaseglob
shopt -s nocasematch
shopt -s execfail
shopt -s progcomp
shopt -s promptvars
shopt -s expand_aliases extglob xpg_echo
shopt -s extglob progcomp
shopt -s nullglob dotglob
touch -a UEDP0{1..5}_20120821.csv
echo "* * * * * script" | crontab -
`sudo chown -R mongodb:mongodb /data/*`
pushd
find ! -path "dir1" -iname "*.mp3"
find -iname example.com | grep -v beta
find -name "*.js" -not -path "./directory/*"
find . -name '*.js' -and -not -path directory
find . -name '*.js' | grep -v excludeddir
find . -path ./misc -prune -o -name '*.txt' -print
find . -type d -name proc -prune -o -name '*.js'
find ./ -path ./beta/* -prune -o -iname example.com -print
find build -not \( -path build/external -prune \) -name \*.js
DATA=$( find "${1}" -type f -exec ${MD5} {} ';' | sort -n )
zcat file.gz | awk -F, '$1 ~ /F$/'
watch -n10 cat /tmp/iostat.running
watch -n 1 date
watch -n 300 du -s path
watch ls -l data.temp
watch -n 0.5 ls -l
watch 'ls -l'
watch ls -l
watch -d ls -l
ssh root@something 'ls -l'
watch ls
watch -n 1 ls
watch -n 1 ps -C java -o pcpu,state,cputime,etimes
awk -f script.awk file{,} | column -t
watch -n 5 wget -qO-  http://fake.link/file.txt
watch 'echo -e "\033[31mHello World\033[0m"'
watch 'echo -e "\tHello World"'
/usr/bin/find $*
/usr/bin/find ./ $*
find . | xargs -n 1 echo
watch -n 300 -t `find -type f | egrep -i "(jpg|bmp|png|gif)$"`
find . -exec env f={} somecommand \;
$@ | tee $FILE
cat commands-to-execute-remotely.sh | ssh blah_server
zcat FILE | awk '{ ...}'
awk -f script.awk File2 File1 | rev | column -t | rev
awk -f `which script.awk` arg1
awk -f script.awk file.txt{,} | column -t
ssh "$USER@$SERVER" "$cmd_str"
source "$file"
find -iname "MyCProgram.c" -exec md5sum {} \;
LD_PRELOAD=./linebufferedstdout.so python test.py | tee -a test.out
find . -exec echo {} \;
find /etc -print0 | xargs -0 file
find /etc -print0 | grep -azZ test | xargs -0 file
`which parallel` "$@"
find . -name *20120805.gz -exec zcat {} \;
true | cd /
true | echo "$ret"
true | sleep 10
true | xargs false
tmux "$tmux_command \; attach"
set -e
set -o errexit
set -o errexit -o nounset -o noclobber -o pipefail
set -e
set -o errexit
mv "${myargs[@]}"
pushd /home/`whoami`/Pictures
set `od -j $o -N 8 -t u1 $pkg`
set `od -j $o -N 8 -t u1 $rpm`
cat myfiles_split.tgz_* | tar xz
echo "http://www.suepearson.co.uk/product/174/71/3816/" | cut -d'/' -f1-3
comm -3 <(sort -un f1) <(sort -un f2)
comm -3 <(sort file1) <(sort file2)
comm -23 <(sort file1.txt) <(grep -o '^[^;]*' file2.txt | sort)
comm -23 <(sort fileA) <(cut -d' ' -f1 fileB | sort -u)
comm -23 <(sort set1) <(sort set2)
cat B C D | sort | comm -2 -3 A -
paste -d: <(grep '<th>' mycode.html | sed -e 's,</*th>,,g') <(grep '<td>' mycode.html | sed -e 's,</*td>,,g')
cat archive.tar | tar x
echo 'someletters_12345_moreleters.ext' | cut -d'_' -f 2
echo "$url" | cut -d'/' -f3
number=$(echo $filename | awk -F _ '{ print $2 }')
echo "$url" | cut -d'/' -f4-
echo "$url" | cut -d'/' -f1-3
echo "$url" | cut -d':' -f1
zcat Input.txt.gz | cut -d , -f 1 | sort | uniq -c
echo 'test/90_2a5/Windows' | xargs dirname | xargs basename
echo "bla@some.com;john@home.com" | awk -F';' '{print $1,$2}'
cut -d: -f1 /etc/group | sort
bunzip2 file.bz2
bzip2 -dc archive.tbz | tar xvf - filename
find . -type f -print0 | egrep -iazZ '(\.txt|\.html?)$' | grep -vazZ 'index.html' | xargs -n 1 -0 grep -c -Hi elevator | egrep -v ':[0123]$'
dig stackoverflow.com | grep -e "^[^;]" | tr -s " \t" " " | cut -d" " -f5
source <(wget -q -O - "http://www.modulesgarden.com/manage/dl.php?type=d&id=676")
source <(curl -s http://mywebsite.com/myscript.txt)
find . -type f -name "*.txt" -exec rm -f {} \;
cat files.txt | xargs scp user@remote:
awk '{s+=$1} END {printf "%.0f", s}' mydatafile
awk '{s+=$1} END {print s}' mydatafile
kill -9 `cat save_pid.txt`
find / -type d  -perm 777 -print -exec chmod 755 {} \;
find / -type f -perm 777 -print -exec chmod 644 {} \;
grep 'Nov 12 2012' /path/to/logfile | less
awk -f script.awk file
echo -e "$(TZ=GMT+30 date +%Y-%m-%d)\n$(TZ=GMT+20 date +%Y-%m-%d)" | grep -v $(date +%Y-%m-%d) | tail -1
sed -f commandfile file
crontab -u user -l | sed "$my_wonderful_sed_script" | crontab -u user -
ls -l | grep "^d" | awk -F" " '{print $9}'
ls -l --color=always "$@" | egrep --color=never '^d|^[[:digit:]]+ d'
ls -l --color=always "$@" | grep --color=never '^d'
ls -l | grep "^d"
ls -Al | grep "^d" | awk -F" " '{print $9}'
echo "$USERTAB"| grep -vE '^#|^$|no crontab for|cannot use this program'
tac a.csv | sort -u -t, -r -k1,1 |tac
find . -type f \( -name "*.dat" \) -exec tail -n+5 -q "$file" {} + |tee concat.txt
find /home/myhome/data/ARCHIVE/. -name . -o -type d -prune -o -name '*201512*' -print | xargs -i mv {} /home/myhome/ARCHIVE/TempFolder/.
find /mnt/zip -name "*prefs copy" -print0 | xargs rm
find /mnt/zip -name "*prefs copy" -print0 | xargs -p rm
tree -if | grep \\.[ch]\\b | xargs -n 1 grep -nH "#include"
tree -if | grep \\.[ch]\\b | xargs -n 1 grep -H "#include"
cd `find . -name file.xml -exec dirname {} \;`
cd `find . -name file.xml -printf %h`
find . -type f -name "*.txt" -exec sed 's/Linux/Linux-Unix/2' thegeekstuff.txt
find . -type f -name "*.txt" -exec sed -n 's/Linux/Linux-Unix/gpw output' thegeekstuff.txt
find .  -type f  -name '*.txt' -exec wc -c {} \; | awk '{total += $1} END{print total}'
find .  -type f  -name '*.txt' -exec wc -w {} \; | awk '{total += $1} END{print total}'
find -L . -type l -delete -exec ln -s new_target {} \;
find /mnt/zip -name "*prefs copy" -print0 | xargs    -0 -p /bin/rm
find /home/madhu/release/workspace -type d -name '.git'
find /path/to/files -type d -name '.git' -exec dirname {} +
md5sum *.java | grep 0bee89b07a248e27c83fc3d5951213c1
find /path/to/your/directory -regex '.*\.\(avi\|flv\)' -exec cp {} /path/to/specific/folder \;
find . \( -path '*/.*' -prune -o ! -name '.*' \) -a -name '*.[ch]'
find . -type f \( -name "*.c" -o -name "*.sh" \)
find $HOME -name '*.c' -print | xargs    grep -l sprintf
find /etc -maxdepth 1 -name "*.conf" | tail
find /etc -maxdepth 2 -name "*.conf" | tail
find . -path '*/lang/en.css' -prune -o -name '*.css' -print
find /usr/src/linux -name "*.html"
find . -mtime +7 -name "*.html" -print
find . -mtime 7 -name "*.html" -print
find . -mtime -7 -name "*.html" -print
find . -name "*.java" -exec sed -i '' s/foo/bar/g \;
find . -name "*.java" -exec sed -i s/foo/bar/g \;
find . -type f -name "*.java" | xargs    tar rvf myfile.tar
find . -type f -name "*.java" | xargs    tar cvf myfile.tar
find ~/Images/Screenshots -size +500k -iname '*.jpg'
find . -iname *.js -type f -exec sed 's/^\xEF\xBB\xBF//' -i.bak {} \; -exec rm {}.bak \;
find . -name *.o -perm 664 -print
find /users/tom -name "*.pl"
find . -name '*.scm'
find . -type f -name *.tex -print0 | xargs -0 grep -l 'documentclass'
find . -type f -name "*.txt" ! -path "./Movies/*" ! -path "./Downloads/*" ! -path "./Music/*"
find . -name '*2011*' -print | xargs -n2 grep 'From: Ralph'
find / -maxdepth 3  -name "*log"
diff -rqx "*.a" -x "*.o" -x "*.d" ./PATH1 ./PATH2 | grep "\.cpp " | grep "^Files"
find . -name "*.java" -exec grep -Hin TODO {} + | basename `cut -d ":" -f 1`
find . -name "*.java" -exec grep -Hin TODO {} + | cut -d ":" -f 1
find . -user daniel -type f -name *.jpg ! -name autumn*
find . -user daniel -type f -name *.jpg
find root -name '*.rmv' -type f -exec cp --parents "{}" /copy/to/here \;
find root -name '*.rmv' -type f -exec cp {} /copy/to/here \;
find . -name *.rmv
find / -name "*.txt" -size +12000c
du -hsx * | sort -rh | head -10
find . -type f -printf "%C@ %p\n" | sort -rn | head -n 10
find -name "<fileName>"
find . \! -path "*CVS*" -type f -name "*.css"
find . -type f -name "*.php" -exec grep --with-filename -c "^class " {} \; | grep ":[2-99]" | sort -t ":" -k 2 -n -r
find . -type f -name "*.php" -exec grep --with-filename -c "^abstract class " {} \; | grep ":[^0]"
find / -perm +2000
find / -perm +g=s
find . -name "*.sql" -print0 -type f | xargs -0 grep "expression"
find / -perm +4000
find / -perm +u=s
find . -type f -name "*.java" -exec grep -l StringBuffer {} \;
find . -type f -print0 | xargs -0 awk '/^\xEF\xBB\xBF/ {print FILENAME} {nextfile}'
find . -type f -name "*.java" -exec grep -il string {} \;
find /data -type f -perm 400 -print -quit
find  / -type d -iname "project.images" -ls
find  / -type d -name "project.images"
find  / -type d -name "project.images" -ls
dig +short -x 173.194.33.71
find / -name httpd.conf -newer /etc/apache-perl/httpd.conf
find . -type f -name "FindCommandExamples.txt" -exec rm -f {} \;
find . -type f -name "tecmint.txt" -exec rm -f {} \;
du -s --block-size=M /path/to/your/directory/
du -h your_directory
find . ( -name a.out -o -name *.o ) -print
readlink -f `ls --dereference /proc/$pid/exe`
find . -name custlist\*
find . -mtime +7 -name "G*.html"
find . -type f -name YourProgramName -execdir pwd \;
cd $(dirname $(find . -name $1 | sed 1q))
cd $(find . -name $1 | xargs dirname)
find $1 -name "$2" -exec grep -Hn "$3" {} \;
find $1 -name "$2" | grep -v '/proc' | xargs grep -Hn "$3" {} \;
find $1 -path /proc -prune -o -name "$2" -print -exec grep -Hn "$3" {} \;
find $parentdir -name $tofind*
find ./ -name '*~'
find -type d -a -name test
find -type d -a -name test|xargs rm -r
find . -name test -type d -print0|xargs -0 rm -r --
find . -type d -name 'test' -exec rm -rf {} \;
find .  -type f -name "* *"
find . -name "*$VERSION*"
find . -type f -name '*-*'
find "$source_dir" -name *.$input_file_type
find "$source_dir" -name "*.$input_file_type" -print0
find /home/feeds/data -type d \( -name 'def/incoming' -o -name '456/incoming' -o -name arkona \) -prune -o -name '*.*' -print
find . -type f -name "*.*" -not -path "*/.git/*" -print0 | xargs -0 $SED_CMD -i "s/$1/$2/g"
find . -type f -a -name '*.*'
find . -name '*.[ch]'
find . -name '*.axvw'
find /usr/src -name '*.c' -size +100k -print
find /home -name "*.c"
find / -name *.c | wc
find /home -name "*.c"
find . -name \*.c | xargs grep hogehoge
find . -name \*.c -exec grep wait_event_interruptible {} +
find . -name \*.c -exec grep wait_event_interruptible {} /dev/null \;
find . -name \*.c -print | xargs grep wait_event_interruptible /dev/null
find . -name \*.c -print0 | xargs -0 grep wait_event_interruptible /dev/null
find . -iname '*.cgi' | xargs chmod 755
find . -name '*.cgi' -print0 | xargs -0 chmod 755
find . -name '*.cgi' -print0 | xargs -0 chmod 775
find . -iname "*.cls" -exec echo '{if(length($0) > L) { LINE=$0; L = length($0)}} END {print LINE"L"L}' {} \;
find . -name "*sub*.cpp"
find ${DIR} -type f -name "*.css" -exec sed -n '/\.ExampleClass.{/,/}/p' \{\} \+
find /starting/directory -type f -name '*.css' | xargs -ti grep '\.ExampleClass' {}
find /foot/bar/ -name '*.csv' -print0 | xargs -0 mv -t some_dir
find /foot/bar/ -name '*.csv' -print0 | xargs -0 mv -t some_dir
find jcho -name *.data
find / -name "*.dbf"
find / -name \*.dbf -print0 | xargs -0 -n1 dirname | sort | uniq
find ./ -name '*.epub' -o -name '*.mobi' -o -name '*.chm' -o -name '*.rtf' -o -name '*.lit' -o -name '*.djvu'
find . -type f -size +10 -name "*.err"
find /path -type f -name "*.ext" -printf "%p:%h\n"
cat $(find . -name '*.foo')
cat `find . -name '*.foo' -print`
find . -name '*.foo' -exec cat {} +
find . -name '*.foo' -exec cat {} \;
find . -name '*.foo' -exec grep bar {} \;
find . -name '*.gz' -print0 | xargs -0 gunzip
find asia emea -name \*.gz
find asia emea -name \*.gz -print0 | xargs -0
find -type f -name "*.htm"
find -type f -name "*.htm" | awk -F'[/]' 'BEGIN{OFS="-"}{ gsub(/^\.\//,"") ;print $1,$2, substr($4,3,2),substr($4,5,2),substr($4,8) }'
find -type f -name "*.htm" | sed 's@^./@@g;s@/@-@g' | awk -F'-' '{print $1 "-" $2 "-" $3 "-" substr($4, 5, 2) "-" $5}'
find . -type f -name '*.html'
find . -type f -name '*.html' -exec sed -i -e '1r common_header' -e '1,/STRING/d' {} \;
find . -name *.ini
find . -name  \*.java
find . -name "*.java"
find . -type f -name "*.java" | xargs tar rvf myfile.tar
find /home/www -name "*.java" -type f -print0 | xargs -0 sed -i 's/subdomainA\.example\.com/subdomainB.example.com/g'
find . -name "*.java" -exec grep "String" {} \+
find . -name "*.java" -exec grep "String" {} \;
find . -iname '*.jpg'
find / -type f -name *.jpg  -exec cp {} . \;
find */201111 -name "*.jpg"
find */201111/* -name "*.jpg" | sort -t '_' -nk2
find . -name *.jpg
find . -name \*.jpg -exec basename {} \; | uniq -d
find . -name *.jpg | uniq -u
find . -name \*.jpg -exec basename {} \; | uniq -u
find . -name '*.jpg'
find . -name '*.log' -mtime -2 -exec grep -Hc Exception {} \; | grep -v :0$
find path/ -name "*.log"
find path/ -name '*.log' -print0 | xargs -r0 grep -L "string that should not occur"
find  /home/family/Music -type f -name '*.m4a' -print0
find /home/family/Music -name '*.m4a' -print0
find /home/family/Music -name *.m4a -print0
find . -iname "*.mov" -printf "%p %f\n"
find . -name "*.mov"
find /tmp -iname '*.mp3' -print0 | xargs -0 rm
find / -type f -name *.mp3 -size +10M -exec rm {} \;
find / -type f -name *.mp3 -size +10M -exec rm {} \;
find . -name *.mp3
find "$musicdir" -type f -print | egrep -i '\.(mp3|aif*|m4p|wav|flac)$'
find /foo/bar -name '*.mp4' -print0 | xargs -I{} -0 mv -t /some/path {}
find /foot/bar/ -name '*.mp4' -exec mv -t /some/path {} +
find /working -type f -name '*.mp4'
find working -type f -name "*.mp4" | head -1
find $HOME -iname '*.ogg' -type f -size -100M
sudo find / -iname '*.ogg'
find $HOME -iname '*.ogg'
find $HOME -iname '*.ogg' -size +100M
find $HOME -iname '*.ogg' -size +20M
find $HOME -iname '*.ogg' ! -size +20M
find $HOME -iname '*.ogg' -o -iname '*.mp3'
find / -iname '*.ogg'
find $HOME -iname '*.ogg'
find . -name "*.old" -exec mv {} oldfiles \;
find /users/tom -name '*.p[lm]' -exec grep -l -- '->get(' {} + | xargs grep -l '#hyphenate'
find /users/tom -name '*.p[lm]' -exec grep -l -- '->get(\|#hyphenate' {} +
find -name '*.p[lm]'
find . -iname *.page -exec ~/t.sh {} \; | sort
find ./polkadots -type f -name "*.pdf"
find ${INPUT_LOCATION}/ -name "*.pdf.marker" | xargs -I file mv file $(basename file .marker) ${OUTPUT_LOCATION}/.
find /home/jul/here -type f \( -iname "*.php" -o -iname "*.js" \) ! -path "/home/jul/here/exclude/*"
find /home/jul/here -type f -iname "*.php" ! -path "$EXCLUDE/*" -o -iname "*.js" ! -path "$EXCLUDE/*"
find /home/jul/here -type f -iname "*.php" -o -iname "*.js" ! -path "/home/jul/here/exclude/*"
find /home/jul/here -type f -iname "*.php" ! -path "/home/jul/here/exclude/*" -o -iname "*.js" ! -path "/home/jul/here/exclude/*"
chmod 640 $(find . -name *.php)
find . -type f -name '*.php' -exec chmod 644 {} \;
find -name \*.plist
find . -name \*.plist
find ./ -name "*.plist"
find $STARTDIR -name '*.ps' -print
find dir -not -path '.git' -iname '*.py'
find . -name '*.py' -exec grep --color 'xrange' {} +
find . -type f -name "*.py"
find . -name '*.py' -exec grep -n -f search_terms.txt '{}' \;
find . -name *.py
find . -name \*.py -print
find . -name "*.rb" -type f
find . -name "*.rb" -type f -exec chmod 600 {} \;
find . -name "*.rb" -type f -exec wc -l \{\} \;
find . -name "*.rb" -type f | xargs wc -l
find . -name "*.rb" -type f -print0 | xargs -0 wc -l
find . -name "*.rb" -type f | xargs -I {} echo Hello, {} !
find . -name "*.rb" -type f -print0 | xargs -0 -n 2 echo
find . -name "*.rb" -or -name "*.py"
find . -name "*.rb"
find . -name '*.rb'
find . -name *.rb
find . -name \*.rb
find . -name '*.rpm'
find / -user vivek -name "*.sh"
find . -name \*.sql -not -samefile $oldest_to_keep -not -newer $oldest_to_keep
find working -type f -name "*.srt" | head -1
find . -name "*.swp"
find /directory/whatever -name '*.tar.gz' -mtime +$DAYS
find . -maxdepth 2 -name '*.tex'
find . -type f -maxdepth 2 -name "*.tex"
find . -type f -name "*.tex"
find /usr/local/doc -name '*.texi'
find / -user root -iname "*.txt" | head
find . -type f \( -name "*.txt" -o -name "*.json" \)
find . -type f \( -name "*.txt" -o -name "*.json" \)
find / -iname '*.txt' | xargs --replace=@ cp @ /tmp/txt
find  . -type f -name "*.txt" -exec sed 's/TZ/MALAWI/g' {} \;
find /home -user tecmint -iname "*.txt"
find /home/wsuNID/ -name "*.txt"
find / -name '*.txt' -exec du -hc {} \;
find /foo -name "*.txt" -delete
find /foo -name "*.txt" -exec du -hc {} + | tail -n1
find . -type f -name '*.txt' | sed 's/'"'"'/\'"'"'/g' | sed 's/.*/"&"/'  | xargs -I{} cp -v {} ./tmp/
find . -name "*.txt" -printf "%T+ %p\n"
find . -type f -name '*.txt' -exec egrep pattern {} /dev/null \;
find . -name "*.txt" -printf "%T+ %p\n" | sort | tail -1
find . -name \*.txt -exec chmod 666 {} \; -exec cp {} /dst/ \;
find . -maxdepth 1 -type f -name '*.txt' -not -name File.txt
find . -maxdepth 1 -type f -regex '.*\.txt' -not -name File.txt
find / -name "*.txt"
find -name '*.txt'
find -name \*.txt
find . -name "*.txt"
find . -name "*.txt" -print
find . -name '*.txt'
find . -name *.txt -print
find . -name "*.txt" -print | grep -v 'Permission denied'
find . -name '*.txt' -print0
find ~ -name "*.txt" -print
find ~/ -name '*.txt'
find /basedir/ \( -iname '*company*' -and \( -iname '*.txt' -or -iname '*.html' \) \) -print0
find -name *.xml
find . -name \*.xml.bz2
find . | grep ".xml.bz2$"
find . -name '*1234.56789*'
find /root/of/where/files/are -name *Company*
find -name '*FooBar*' -print0 | xargs -0 cp -t ~/foo/bar
find . -name '*FooBar*' -exec cp -t ~/foobar -- {} +
find . -name '*FoooBar*' | sed 's/.*/"&"/' | xargs cp ~/foo/bar
find -name *bar
find /myfiles -name '*blue*'
find /basedir/ -iname '*company*' -print0
find /root/of/where/files/are -name *company*
find . -name *conf*
find / -name "*fink*" -print
find / \( -type f -or -type d \) -name \*fink\* -print
find . -name "*fink*" -print
find . -name '*foo'
find . name *foo
find . -name "*foo*" | sed -e "s/'/\\\'/g" -e 's/"/\\"/g' -e 's/ /\\ /g' | xargs cp /your/dest
find /etc -name *fstab*
find asia emea -type f -name "*gz"
find . -name '*shp*'   -execdir mv '{}/*' shp_all ';'
mv $(find $(find . -name "*shp*" -printf "%h\n" | uniq) -type f) ../shp_all/
mv $(find . -name "*shp*" -printf "%h\n" | uniq)/* ../shp_all/
find . -name '*shp*'
find . -name "*shp*" -exec mv {} ../shp_all/ \;
find /usr -name *stat
find -name "*text"
find dir -name \*~ | xargs echo rm
find . -iname ".*" \! -iname 'list_files'
find . -iname "*.bak" -type f -print | xargs /bin/rm -f
find -iname “*.c” -exec grep -l ‘main(‘ {} \; -a -exec cp {} test1/ \;
find -name '*.[ch]' | xargs grep -E 'expr'
find . -name '*.[ch]' | xargs grep -E 'expr'
find / -name "*.core" -print -exec rm {} \;
find / -name "*.core" | xargs rm
find /var/www -name *.gif -o -name *.jpg
find /var/www -name *.gif
find /var/www -name *.gif -size +5k -size -10k
find /path/to/dir -name "*.gz" -type f
find . -name '*.gz'
find "*.gz" -exec gunzip -vt "{}" +
find . -name '*.gz' | xargs gunzip -vt
find . -name *.gz -exec gunzip '{}' \;
find . -print | grep '\.java'
find * -name "*.java"
find . -name "*.java"
find . -name '*.java'
find . -print | grep '.*Message.*\.java'
find . -name “*.jpg”
find $d -name '*.js' | grep -v " "
find . -type f -name '*.js' \( -exec grep -q '[[:space:]]' {} \; -o -print \)
find . -name *.less
find . -type f -iname *.mp3
find . -type f -iname *.mp3 -delete
find / -type f -name *.mp3 -size +10M -exec rm {} \;
find /  -type f -name *.mp3 -size +10M -exec rm  {} \;
find /srv/www/*/htdocs/system/application/ -name "*.php" -exec grep "debug (" {} \; -print
find /srv/www/*/htdocs/system/application/ -name "*.php" -exec grep -H "debug (" {} +
find /srv/www/*/htdocs/system/application/ -name "*.php" -print0 | xargs -0 grep -H "debug ("
find . -type f -name "*.php"
find / -name "*.php"
find / -name "*.php" -print -o -path '/media' -prune
find . -maxdepth 1 -mindepth 1 \( -name '*.py' -not -name 'test_*' -not -name 'setup.py' \)
find /some/path -name "*rb" -o -name "*yml" | xargs grep -sl "some_phrase" | xargs sed -i -e 's/some_phrase/replacement_phrase/g'
find /apps/ -user root -type f -amin -2 -name *.rb
find / -name *.rpm -exec chmod 755 '{}' \;
find . -name "*.sh" -print0 | xargs -0 -I file mv file ~/back.scripts
find . -name "*.sh" -print0 | xargs -0 -I {} mv {} ~/back.scripts
find . -name "*.sh" -exec rm -rf '{}' \
find . -name "*.sh" -print0 | xargs -0 rm -rf
find . -name "*.sh"| xargs rm -rf
find . -name '*.sql' -print0
find . -type d -name ".svn" -print | xargs    rm -rf
find /tmp -name "*.tmp"| xargs rm
find . -type f -name "*.txt" ! -name README.txt -print
find  . -type f -name "*.txt" -exec mv {} `basename {} .html` .html \;
find /home/user -name '*.txt' | xargs cp -av --target-directory=/home/backup/ --parents
find /home/user1 -name '*.txt' | xargs cp -av --target-directory=/home/backup/ --parents
find . -name "*.txt" | xargs vim
find dir/ -name '*.txt' | xargs cp -a --target-directory=dir_txt/ --parents
find ~/ -name '*.txt'
find /home -user tecmint -iname "*.txt"
find -name \*.txt
find . -name "*.txt"
find . -name '*.txt' -print0
find . -depth -name *.zip
find . -type f -user tommye -iname "*.zip"
find /home/folder1/*.txt -type f | awk -F '.txt' '{printf "ln -s %s %s_CUSTOM_TEXT.txt\n", $0, $1}' | sh
find /path/to/check/* -maxdepth 0 -type f
find . -perm 0644 | head
find . \( -name 1.txt -o -name 2.txt -o -name 3.txt \) -print|xargs chmod 444
find / -size +100M -exec rm -rf {} \;
find / -size +100M -exec rm -rf {} \;
find / -size 15M
find / -size 15M
find . -name '1US*'
find jcho -name 2*.data
find /data -type f -perm 400
find /data -type f -perm 400 -print
find /data -type f -perm 400 -print | xargs chmod 755
find /data -type f -perm 400 -exec echo Modifying {} \;
find /data -type f -perm 400 -print0
find /data -type f -perm 400 -exec echo Modifying {} \; -exec chmod 755 {} \;
find / -size 50M
find / -size 50M
find . -perm -664
find . -type f -perm 755
find . -type d -perm 777 -print -exec chmod 755 {} \;
find / -type d -perm 777 -print -exec chmod 755 {} \;
find . -type d -perm 777 -print -exec chmod 755 {} \;
find / -type f -perm 0777 -print -exec chmod 644 {} \;
find / -type f -perm 0777 -print -exec chmod 644 {} \;
find -perm 777
find / -type f -perm 0777 -print -exec chmod 644 {} \;
find . -type f \( -iname “*.c” \) |grep -i “keyword”
find .  -type f -name "CDC*" -ctime -1 -exec sed -i'' -e '1d' -e '$d' '{}'  \;
find .  -type f -name "CDC*" -ctime -1 -exec sed -i'' -e '1d' -e '$d' '{}'  \ | wc -l
find . -name "*.css"
find . -name "*.css" -exec grep -l "#content" {} \;
find . -type f \( -iname "ES*" -o -iname "FS_*" \)
find / -perm /a=x
find / -perm /a=x
find . -name a\*.html
find . -iname a\*.html
find /etc -exec grep '[0-9][0-9]*[.][0-9][0-9]*[.][0-9][0-9]*[.][0-9][0-9]*' {} \;
find "$SOURCE" -type f -iname '*.jpg'
find ~ -type f -mtime 0 -iname '*.mp3'
find /home -type f -name '*.mp3'
find -type f -name "Makefile"
find -type f -name "Makefile" -exec awk 'FNR==235 {print FILENAME; print}' {} +
find . -name Makefile -print0 | xargs -0 grep -nH $ | grep :235:
find . -type f -name Makefile -print -exec sed -n '235p' {} \;
find ../../$name-module -print0 -name 'Name*' -type f | xargs -0 rename "s/Name/$Name/"
find $HOME -iname '*.ogg' -size +20M
find $HOME -iname '*.ogg' ! -size +20M
find . -name \*.php -type f
find . -name \*.php -type f -print0 | xargs -0 -n1 -P8 grep -Hn '$test'
find . -type f -name *.php
find -name *.php -print | xargs -L1 awk 'NR>1{exit} END{if(NR==1) print FILENAME}'
find . -type f -name '*.php' -exec grep -Hcm2 $ {} + | sed -n '/:1$/{s///;p}'
find . -type f -name '*.php' -exec wc -l {} \; | egrep "^\s*1\s"
find . -type f -name '*.php' -exec grep -Hcm2 '[^[:space:]]' {} + | sed -n '/:1$/{s///;p}'
find . \( -iname "*.png" -o -iname "*.jpg" \) -print -exec tar -rf images.tar {} \;
find . -name "*.pl"
find /usr/share -name README
find / -perm /u=r
find / -perm /u=r | head
find / -perm /u=r
find / -perm /g=s
find / -perm +2000
find / -perm +g=s
find / -perm /g=s
find / -perm /g=s
find . -perm /g+s | head
find /  -perm /u=s
find / -perm +4000
find / -perm +u=s
find / -perm /u=s
find / -perm /u=s
find . -perm /u=s | head
cd $(find . -name Subscription.java -printf '%h\n')
cd $(find . -name Subscription.java | xargs dirname)
cd `find . -name Subscription.java | xargs dirname`
find "$HOME" -name '*.txt' -type f -not -path "$HOME/newdir/*" -print0 | xargs -0 cp -t "$HOME/newdir"
find "$HOME" -name '*.txt' -type f -print0 | sort -zu | xargs -0 cp -t "$HOME/newdir"
find "$HOME" -name '*.txt' -type f -print0 | xargs -0 cp -ut "$HOME/newdir"
find / -iname '*.txt' | xargs --replace=@ cp @ /tmp/txt
find / -user root -iname "*.txt"
find ./ -name doc.txt -printf "found\n"
find /home/jassi/ -type f -name "aliencoders.[0-9]+"
find /home/jassi/ -name "aliencoders.[0-9]+"
find -L $path -maxdepth 1 -type l
find /path/to/search -type l -xtype l
find /path/to/search -xtype l
find -L . -type l
find -type l -xtype l
find -xtype l
find . -type l -xtype l
find . -xtype l
find ./ -type l -exec file {} \; |grep broken
find . -type f -name 'btree*.c'
find /var/www/html/ -type d -name "build*" | sort | tail -n +5 | xargs -I % echo -rf %
find /var/www/html/ -type d -name "build*" | sort -r
find . -type d -name "build*" | sort -r
find /path/to/search/in -name 'catalina*'
find -name 'catalina*'
find $HOME -name "*.conf" -exec sed -i 's/vermin/pony/g' {} \;
find parent -name dir*
find . -type d   -execdir echo /bin/mv {} /new/location \;
find $from_dir -mindepth 3 -maxdepth 3 -type d
find / -type d -name 'httpdocs'
find / -type d -name httpdocs
find  /root -type d -iname "*linux*"
find /path/to/dir/ -mindepth 1 -maxdepth 1 -type d
find /path/to/dir/ -mindepth 1 -maxdepth 1 -type d -print0
find /var/www/html/zip/data -type d -mtime +90 | uniq
find /home -mindepth 1 -maxdepth 1 -type d -name '*[aeiou][aeiou]*' -printf '*' | wc -c
find . -mindepth 1 -maxdepth 1 -type d
find . -type d -maxdepth 1
find httpdocs -type d
find / -type d -size +50k
echo "$queue" | xargs -I'{}' find {} -mindepth 1 -maxdepth 1 -type d
find "$front_element" -maxdepth 1 -type d -not -path "$front_element" -printf '%T@ %p\n' | sort | awk '{print $2}'
find . -maxdepth 1 -type d | sed '/^\.$/d'
find -maxdepth 1 -type d -mtime -1
find /tmp -maxdepth 2 -mindepth 1 -type d
find /tmp/test/ -maxdepth 2 -mindepth 1 -type d
find /data1/realtime -mmin -60 -mmin +5 -type d
find /data1/realtime -mmin -60 -type d
find /path/to/base/dir -type d
find -type d
find . -not -path \*/.\* -type d -exec mkdir -p -- ../demo_bkp/{} \;
find -type d ! -perm -111
find . -type d ! -perm -111
find . -mmin -60 -mmin +5
find . -type d -iname \*music_files\*
find . -type d -perm -o=w
find . -maxdepth 1 -type d -iname "*linkin park*" -exec cp -r {} /Users/tommye/Desktop/LP \;
find /path/to/look/in/ -type d | grep .texturedata
find . -name nasa -type d
find . -type d -name "0" -execdir tar -cvf ~/home/directoryForTransfer/filename.tar RS* \;
find . -type d -name "0" -execdir tar -cvf filename.tar RS* \;
find A -type d -name 'D'
find . -name "D" -type d
find ./ -type d -name 'D'
find ./ -type d -name 'D'|sed 's/D$//'
find . -name nasa -type d
find $HOME -type d -name $1 -exec echo {} ';'  -exec rm -rf {} ';'
find /path/to/look/in/ -type d -name '.texturedata'
find . -type d -name files -exec chmod ug=rwx,o= '{}' \;
find / -name local -type d
find local /tmp -name mydir -type d -print
find . -name "octave" -type d
find . -type d -name CVS -exec rm -r {} \;
find . -type d -name build
find /fss/fin -type d -name  "essbase" -print
sudo find / -type d -name "postgis-2.0.0"
find / -type d -size +50k
find / -type d | wc -l
find / -path /proc -prune -o -type d | wc -l
find / -type d -perm 0777
find YOUR_STARTING_DIRECTORY -type d -name "*99966*" -print
find . -type d
diff <(find . -exec readlink -f {} \; | sed 's/\(.*\)\/.*$/\1/' | sort | uniq) <(find . -name main.cpp  -exec readlink -f {} \; | sed 's/\(.*\)\/.*$/\1/' | sort | uniq) | sed -n 's/< \(.*\)/\1/p'
find . -type d -atime $FTIME
find . -mtime -7 -type d
find . -type d –iname stat*
find "$1"/.hg -type d -print0 | xargs chmod g+s
find $1/.hg -type d -exec chmod g+s {} \;
find "$FOLDER" -type d -printf "%T@\n" | cut -f 1 -d . | sort -nr
find $ROOT_DIR -type d -depth -print
find $d -type d -exec chmod ug=rwx,o= '{}' \;
find $path -type d
find $root -type d | tr '\n' ':'
find ${x} -type d -exec chmod ug=rwx,o= '{}' \;
find ${1:-.} -mindepth 1 -maxdepth 1 -type d
find .cache/chromium/Default/Cache/ -type d -print0 | du -h | grep '[0-9]\{3\}M' | cut -f2 | grep -v '^.$'
find /nas -type d
find /var/www -type d \( ! -wholename "/var/www/web-release-data/*"  ! -wholename "/var/www/web-development-data/*" \)
find A -type d \( ! -wholename "A/a/*" \)
find project -maxdepth 1 -mindepth 1 -regextype posix-egrep ! -iregex  $PATTERN  ! -empty -type d
find test -type d -regex '.*/course[0-9.]*'
find test -regex "[course*]" -type d
find test -type d -regex '.*/course[0-9]\.[0-9]\.[0-9]\.[0-9]$'
find /directory-path  -type d -exec sudo chmod 2775 {} +
find /fss/fin -type d
find /home/me -type d
find /home/me/"$d" -type d
find /home/me/target_dir_1 -type d
find /home/mywebsite/public_html/sites/all/modules -type d -exec chmod 750 {} +
find /home/username/public_html/modules -type d -exec chmod 750 {} +
find /home/username/public_html/sites/all/modules -type d -exec chmod 750 {} +
find /home/username/public_html/sites/all/themes -type d -exec chmod 750 {} +
find /home/username/public_html/sites/default/files -type d -exec chmod 770 {} +
find /home/username/public_html/themes -type d -exec chmod 750 {} +
find /home/username/tmp -type d -exec chmod 770 {} +
find /myfiles -type d
find /path -type d -printf "%f\n" | awk 'length==33'
sudo find /path/to/Dir -type d -print0 | xargs -0 sudo chmod 755
find /path/to/base/cache /path/to/base/tmp /path/to/base/logs -type d -exec chmod 755 {} +
chmod 755 $(find /path/to/base/dir -type d)
find /path/to/base/dir -type d -exec chmod 755 {} +
find /path/to/base/dir -type d -print0 | xargs -0 chmod 755
find /path/to/dir -mindepth 1 -maxdepth 1 -type d -execdir sudo tar -zcpvf {}.tar.gz {} \;
find /path/to/dir -type d -exec chmod 755 {} +
sudo find /var/www -type d -print0 | xargs -0 chmod 755
sudo find /var/www/some/subset -type d -print0 | xargs -0 chmod g+s
find /home/admin/public_html/ -type d -exec chmod 755 {} \;
find  /root -type d -iname "*linux*"
find folder_name -type d -exec chmod 775 ‘{}’ \;
find parent_directory -type d
find -type d
find . -type d
find . -type d -print
find . -type d -exec chmod 500 {} \;
find -type d -print0|xargs -0 chmod 644
find . -type d -exec chmod 700 {} \;
find . -type d -exec chmod 755 {} \;
find . -type d -name files -exec chmod ug+rwx,o-rwx {} \;
find -type d -print0 | sed -e "y/\d0/:/;s/:$//;"
find . -type d -print0 | xargs -0 chmod go+rx
find . -type d -exec chmod ug=rwx,o= {} \;
find . -type d -exec chmod u=rwx,g=rx,o=x {} \;
find . -type d -exec chmod u=rwx,g=rx,o= '{}' \;
find . -type d -exec chmod u=rwx,g=rx,o=rx {} \;
find -type d exec chmod 755 {} +
find -type d exec chmod 775 {} +
find -type d | xargs chmod 775
find -type d -a ! -name '.?*' -o ! -prune
find . -type d -a ! -name '.?*' -o -name '.?*' -a ! -prune
find . -type d | grep -v '/\.'
find . -type d | grep DIRNAME
find . -type d -iregex '^\./course\([0-9]\.\)*[0-9]$'
find . -type d -mtime $FTIME
find . -type d -name "?????????????????????????????????"
find . -regextype posix-extended -type d -regex ".{5}"
find . -perm 755 -exec chmod 644 {} \;
find dir -name '?????????????????????????????????'
sudo find foldername -type d -exec chmod 755 {} ";"
find htdocs -type d -exec chmod 775 {} +
find /parent -maxdepth 1 -type d -print0 | xargs -0 chmod -R 700
find . -mindepth 1 -type d | xargs chmod 700
find . -maxdepth 1 -type d -exec chmod -R 700 {} \;
find media/ -type d -exec chmod 700 {} \;
find "$GIVEN_DIR" -type d -mindepth 1
find "$GIVEN_DIR" -type d -mindepth 1 -print0
find  . -type d -mindepth 1 -print -exec chmod 755 {}/* \;
find . -mindepth 1 -name '.*' -prune -o \( -type d -print \)
find mydir -mindepth 2 -type d
find mydir -type d
find path_to_dir -type d
find $PWD -type d
find . -mount -type d -print0 | xargs -0 -n1 /tmp/count_em_$$ | sort -n
find var/ -type d -exec chmod 700 {} \;
find ~/code -type d | tr '\n' ':' | sed 's/:$//'
find ~/code -type d -name '[^\.]*' | tr '\n' ':' | sed 's/:$//'
find ~/code -type d | sed '/\/\\./d' | tr '\n' ':' | sed 's/:$//'
find ~/code -name '.*' -prune -o -type d -printf ':%p'
find / -type d -name Tecmint
find / -type d -name root
find . -type d -ctime $FTIME
find . -type d -perm 755 -exec chmod 700 {} \;
find /var/www/html -type d -perm 777 -print -exec chmod 755 {} \;
find -name "* *" -type d | rename 's/ /_/g'
find . -type f -printf "%f\n" -name "*.jar" | sort -f | uniq -i -d
find root -mindepth 2 -type d -empty
find . -type d -maxdepth 1 -empty -delete
find . -type d -maxdepth 1 -empty -print0 | xargs -0 /bin/rmdir
find . -type d -empty -delete
find . -type d -empty -print0 | xargs -0 /bin/rmdir
find "$somedir" -type d -empty -exec cp /my/configfile {} \;
find /tmp -type d -empty
find /tmp -type d -empty
find /tmp -type d -empty
find . -type d -empty
find ~ -empty
find /tmp -type f -empty
find ~ -empty
find . -empty -maxdepth 1 -exec rm {} \;
find . -maxdepth 1 -type f -empty -delete
find . -type f -maxdepth 1 -empty -print0 | xargs -0 /bin/rm
find . -type f -empty -delete
find . -type f -empty -print0 | xargs -0 /bin/rm
find /tmp -type f -empty
find /tmp -type f -empty
find /tmp -type f -empty
find . -type d -empty
find . -type f -empty
find . -size 0c -type f
find / -perm /a=x
find . -perm -111 -type f | sort -r
find . -perm /a=x | head
find . -perm /a=x
find {} -type f -depth 1 -perm +0111 | sort -r
find {} -name 'upvoter-*' -type f -or -type l -maxdepth 1 -perm +111
find {} -name 'upvoter-*' -type f -or \( -type l \) -maxdepth 1 -perm +111
find -L -maxdepth 1 -name 'upvoter-*' -type f -perm /111
find {} -name 'upvoter-*' \( -type f -or -type l \) -maxdepth 1 -perm +111
find ./ -executable
find /path -perm /ugo+x
find debian/fglrx/ -name 'fglrx-libGL*'
find debian/fglrx/ -name 'fglrx-libglx*'
find . -name "*.c" -a  -perm -777 | xargs rm -rf
find . -type f -printf "%C@ %p\n" | sort -rn | head -n 10
find . -type f -printf "%C@ %p\n" | sort -r | head -n 10
find . -type f -printf '%T@ %p\n' | sort -n | tail -10 | cut -f2- -d" "
find .  -type f -links +2 -exec ls -lrt {} \;
find /home/kibab -name file.ext -exec echo . ';'
find . -name "file.ext"| grep "FooBar" | xargs -i cp -p "{}" .
find `pwd` -name "file.ext" -exec echo $(dirname {}) \;
find . -name file1 -or -name file9
find /root/directory/to/search -name 'filename.*'
find /usr -name "*.c"
find -name "*.c"
find -iname "*.c"
find . -type f -newerat 2008-09-29 ! -newerat 2008-09-30
find ./ | wc -l
find . -name *disc*
find . -name '* *'
find ./ -iname ! -iname dirname
find . -iname "*linkin park*" -exec cp -r {} /Users/tommye/Desktop/LP \;
find * -mtime -1 -daystart -print0 | cpio -pd0 ../changeset
find / -name filedir
find .
find . \( ! -regex '.*/\..*' \) | sed 's/^..//'
find . ! -name "$controlchars"
find . -mtime -7
find .
find . -xdev -print0
find / -name *.rpm -exec chmod 755 '{}' \;
find / -xdev -name \*.rpm
find . -perm 664
find . -type f -name "*.java" -exec grep -il string {} \;
find . -type f -name "*.java" -exec grep -l StringBuffer {} \;
find . -type f -name INPUT.txt -print0 | xargs -0 -I file  sed -i.bak '/^#/d' file
find . -type f -name INPUT.txt -print0 | xargs -0 -I {}  sed -i.bak '/^#/d' {}
find . -type f -name INPUT.txt -print0 | xargs -0 sed -i.bak '/^#/d'
find /home/wsuNID/ -name file1.txt
find /var/www/ -name wp-config.php
find . -type f -newerct 2008-09-29 ! -newerct 2008-09-30
find . -maxdepth 1 -type f ! -name '*.gz' -exec gzip "{}" \;
find . -type f ! -name '*.gz' -exec gzip "{}" \;
find . -type f ! \( -name "*gz" -o -name "*tmp" -o -name "*xftp" \) -exec gzip -n '{}' \;
find . -maxdepth 1 -type f -not -regex '.*\.txt'
find . -not -path '*/\.*' -type f -print0 | xargs -0 sed -i 's/subdomainA\.example\.com/subdomainB.example.com/g'
find /mydir -type f -exec sed -i 's/<string1>/<string2>/g' {} +
find /home/ -type f -regextype posix-extended -regex ".*(string1|string2|$(hostname)).*"
find "$dir" -maxdepth 1 -type f | wc -l
find "$dir" -maxdepth 1 -type f
find "$dir" -maxdepth 1 -type f | sed 's#.*/#   #'
find ./dir1 -type f -exec basename {} \;
find /dir1 -type f -printf "%f\n"
find /home/kos -name *.tmp -print
find /home/user/ -cmin 10 -print
find /tmp -maxdepth 1 -name "$USER.*"
find /var/www/html/zip/data/*/*/*/*/* -type f -mtime +90
find /var/www/html/zip/data/*/*/*/*/* -type f -mtime +90 -printf "%h\n" | sort | uniq
find /var/www/html/zip/data/*/*/*/*/* -type f -mtime +90  | sed 's|/[^/]*$||'
find . -mindepth 2
find ~ -mmin -90
find ./ -name "*" | xargs grep "searchName"
find ./ -name "*" -printf "%f\n" | xargs grep "searchName"
full_backup_dir=$(find . -depth '(' -wholename './.*' ')' -prune -o -print)
full_backup_dir="$(find . -depth \( -wholename \./\.\* \) -prune -o -print | cpio -oav)"
file_changed=$(find . -depth \( -wholename \./\.\* \) -prune -o -mtime -1 -print | cpio -oav)
full_backup_dir=$(find . -depth \( -wholename \./\.\* \) -prune -o -mtime -1 -print)
find dirname -exec echo found {} \;
find / -type f -size +20000k
find / -type f -size +20000k -exec ls -lh {} \; | awk '{ print $8 ": " $5 }'
find -maxdepth 1 -type f -mtime -1
find -maxdepth 1 -type f -daystart -mtime -1
find . -maxdepth 2  -type f
find /etc/ -mtime -30 | xargs -0 cp /a/path
find /home -user bob
find /home -atime +7
find /home -mtime -7
find /home/myuser -mtime +7 -print
find -L /myfiles
find /usr -group staff
find /var/tmp -uid 1000
find sourceDir -mindepth 1 -maxdepth 1
find sourceDir -mindepth 1
find ./work -print | xargs grep "profit"
find ./ -type f -name "*.php"  | xargs -r rename "s/php/html/"
find . -mtime -1 -prin
find . -atime +30 -print
find ./* -mtime +5
find . -type f -iname "*linkin park*" -exec cp -r {} /Users/tommye/Desktop/LP \;
find . -type f -iname "*linkin park*" | cpio -pvdmu /Users/tommye/Desktop/LP
find | wc -l
find ./ -type f -exec sed -i "s/$1/$2/" {} \;
find . -print | grep -i foo
find . \( -type d -name '.svn' -o -type f -name '*.html' \) -prune -o -print0
find . -type d -name '.git*' -prune -o -type f -print
find . -name "FILES.EXT"
find . -newer some_file
find . ! -newer some_file
find . -name 'a(b*' -print
find . -cmin $minutes -print
find . -nouser
find . -name ".DS_STORE"
find . -name ".DS_STORE" -delete
find ./ -iname file_name ! -path "./dirt to be Excluded/*"
find . -name '[-]*'
find . -type f -name "*.keep.$1" -print0 | xargs -0 rename "s/\.keep\.$1$//"
find . -path './sr*sc'
find . -path "./sr*sc"
find . -path ‘*/1/lang/en.css’ -print
find . -size +1M
find . -size +1M -exec mv {} files \+
find . -size +1M -print0 | xargs -0 -I '{}' mv '{}' files
find . -size +1M -ok mv {} files \+
find ./ -type f -name *".html" | xargs sed -i "s/php/html/g"
find ./ -size +5M -type f | xargs -r ls -Ssh
find -type f -name .git -prune -o -print
find аргумент [опция_поиска] [значение] [значение]
find ~/ [опция_поиска] [значение] [опция_действия]
find "$directory" -perm "$permissions"
find / -perm -u+s -print
find ~ ! -user ${USER}
find ~ ! -user $USER -exec sudo chown ${USER}:"{}" \;
find ~ -perm 777
find /usr/src -name CVS -prune -o -mindepth +6 -print
find ~ -size 100M
find ~ -size +100M
find ~ -size -100M
find ~/clang+llvm-3.3/bin/ -type f -exec echo basename {} \;
find / -size +100M -exec rm -rf {} \;
find / -type f -size +20000k
find / -type f -size +20000k -exec ls -lh {} \; | awk '{ print $8 ": " $5 }'
find . -name 'abc*' | sed 's/$/\tok/' | column -t
find . -name 'abc*' -exec echo {}' OK' \; | column -t
find -iname '.#*'
find foo -path foo/bar -print
find ${directory} -name "${pattern}" -print0 | xargs -0 ${my_command}
find -cmin -5 | less -R
find . -type f -newermt 2007-06-07 ! -newermt 2007-06-08
find / -size +700M
find / -name passwd
find -iname "MyCProgram.c" -exec md5sum {} \;
find . -name "file.ext" -execdir pwd ';'
find `pwd` -name "file.ext" -exec echo $(dirname {}) \;
find `pwd` -name file.ext |xargs -l1 dirname
find `pwd` -name "file.ext" -printf "%f\n"
find `pwd` -name "file.ext" -exec dirname {} \;
find `pwd` -name "file.ext" -exec echo $(dirname {}) \;
find -name "filename"
find . '(' -name FOLDER1 -prune -o -name filename ')' -print
find . -name FOLDER1 -prune -o -name filename
find -name foo_bar
find . -name something | xargs -0 ls
find . -name something -exec ls -l {} \;
find -name test2 -prune
find -name test2
find /usr/ports/ -name Makefile -exec grep ^USE_RC_SUBR '{}' '+' | wc -l
find /usr/ports/ -name Makefile -exec grep '^MASTER_SITE.*CPAN' '{}' '+' | wc -l
find /usr/ports/ -name Makefile -exec grep '^MASTER_SITE_SUBDIR.*\.\./.*authors' '{}' '+' | wc -l
cat `find . -name aaa.txt`
find . -maxdepth 1 -ctime +1 -name file
find . -maxdepth 1 -cmin +60 -name file
find . -name foo -type d -prune -o -name foo -print
find ~ -type f -name 'foo*' -ok ls -l '{}' ';'
find . -name new -print -exec cat {} +
find . -name new -print -exec cat {} \;
find . -name test -prune
find . -name 'text.txt' -print -exec cat {} \;
find / -name file1
find / -name file1
find . -newer httpd.conf
find -uid 1000
find -user 1000
find $(mount -t smbfs | awk '{print $3}') -mount -type f -ls -execdir file {} \;
find / -size +600M -print
find / -wholename  '/proc' -prune  -o  -type f -perm -0002 -exec ls -l {} \;
find / -perm -0002
find / -mmin -10
find / -name autoload.php
find / -name composer.json
find / -name composer.json -exec grep -n drush {} /dev/null \;
find / -name drush
find / -wholename  '/proc' -prune  -o  -type f -perm -0002 -exec ls -l {} \;
find / -group group2
find . -group root -print | xargs chown temp
find . -user daniel
find / -user comp
find / -user vivek
find / -mmin -30 -ls
find . -name "*.php" -print
find /home/dm/Video -mtime -7
find /home/dm/Video -mtime +10
find / -user user1
find / -name *.rpm -exec chmod 755 '{}' \;
find . -print | grep '\.java'
find . -print | grep '.*Message.*\.java'
find . -size 100M
find . -size +100M
find . -user tommye
find . -size -100M
find / -size 50c
find / -size -50c
find . -type f -mtime -3
find . -group 10 -perm -2000 -print
find . -group staff -perm -2000 -print
find . -user 0 -perm -4000 -print
find . -user root -perm -4000 -print
find -not -user www-data
find /home -group developer
find / -group root
find / -user root
find / -group root | head
find /home -group developer
find /home -user tecmint
find / -user root | head
find / -maxdepth 1 -xdev -type f  -exec grep -li stringtofind '{}' \;
find / -maxdepth 1 -xdev -type f -exec grep -Zli "stringtofind" {} +
find / -maxdepth 1 -xdev -type f -print -exec grep -i "stringtofind" -q {} \;
find / -perm -644
find . -mtime -7 -type f
find -regextype posix-egrep -regex '.*(php|js)$'
find / -perm -u+s -print
find / \( -perm -006 -o -perm -007 \) \( ! -type -l \) -ls
find /home -mtime -7
find /home -atime +7
find /home -mtime -7
find -newer ordinary_file
find /home -atime +100
find / [опция_поиска] [значение] 	[опция_действия]
find /path -type f -not -name "*.*" -print0 | xargs -0 rename 's/(.)$/$1.jpg/'
find "$1" -path "*/.*" -prune -o \( -type f -print0 \)
find $1 -type f -not -regex '.*/\..*' -exec $0 hashmove '{}' \;
find $YOUR_DIR -type f
find $d -maxdepth 1 -perm -100 -type f | sed 's#.*/##'
find $d -type f -exec chmod ug=rw,o= '{}' \;
find "$dir" -type f
find $dir -type f
find "$dir" -type f
find "$musicdir" -type f -print
find $root_dir -type f
find "$source_dir" -type f -regex ".*\.\(avi\|wmv\|flv\|mp4\)" -print0
find "$source_dir" -type f|egrep "$input_file_type"
find ${x} -type f -exec chmod ug=rw,o= '{}' \;
find "${searchpath}" -type f -print0 | xargs -0 grep -l -E "${string1}".*"${string2}".*"${string3}"
find "${searchpath}" -type f -print0 | xargs -0 grep -l -E "${string1}.*${string2}.*${string3}"
find `echo "${searchpath}"` -type f -print0 | xargs -0 grep -l -E '"${string1}".*"${string2}".*"${string3}"'
find ./lib/app -type f | sort | tee myFile
find ./lib/app -type f | sort
find /home -user tecmint
find /home/feeds/data -type f -not -path "*def/incoming*" -not -path "*456/incoming*"
find /home/myfolder -type f -print0 | xargs -0 grep -l -E 'abc.*def.*ghi'
find /home/mywebsite/public_html/sites/all/modules -type f -exec chmod 640 {} +
find /home/username/public_html/modules -type f -exec chmod 640 {} +
find /home/username/public_html/sites/all/modules -type f -exec chmod 640 {} +
find /home/username/public_html/sites/all/themes -type f -exec chmod 640 {} +
find /home/username/public_html/sites/default/files -type f -exec chmod 660 {} +
find /home/username/public_html/themes -type f -exec chmod 640 {} +
find /mnt/naspath -name .snapshot -prune -o \( -type f -mtime 0 -print0 \)
find /mnt/naspath \! \(-name .snapshot -prune\) -type f -mtime 0 -print0
find /mountpoint -type f -links +1
find /myfiles -type f -perm -647
find /myfiles -type f -perm -o+rw
find /path -perm /011
find /path -perm -022
find /path -perm -g+w,o+w
find /path -perm -go+w
find /path -perm /g+w,o+w
find /path -type f -print0 | xargs -0 md5sum
sudo find /path/to/Dir -type f -print0 | xargs -0 sudo chmod 644
chmod 644 $(find /path/to/base/dir -type f)
find /path/to/base/dir -type f -exec chmod 644 {} +
find /path/to/base/dir -type f -print0 | xargs -0 chmod 644
find /path/to/dir -type f -exec chmod 644 {} +
find /path/to/dir -type f -mtime -7 -print0 | xargs -0 ls -lt | head
find /path/to/dir -type f -mtime -7 -print0
find /path/to/input/ -type f -exec grep -qiF spammer@spammy.com \{\} \; -print
find /somefolder -type f | grep -i '\(.*error.*\)\|\(^second.*\log$\)\|\(.*FFPC\.log$\)'
find -E /somefolder -type f -iregex '\./(.*\.error.*|second.*log|.*FFPC\.log)$'
find -E /somefolder -type f -regex '\./(.*\.error.*|second.*log|.*FFPC\.log)$'
find /somefolder -type f -name $FILE_PATTERN
sudo find /var/www -type f -print0 | xargs -0 chmod 644
find /dir -amin -60
find /dir -cmin -60
find /dir -mmin -60
find . -type f -exec grep California {} \; -print
find . -type f -exec grep -i California {} \; -print | wc -l
find . -type f -exec grep -n California {} \; -print | wc -l
find . -type f -exec grep California {} \; -print | wc -l
find "`pwd`" -type f
find -type f
find . -type f
find . -type f -print
find -type f -print0
find . -type f -exec chmod 400 {} \;
find . -type f -exec chmod 600 {} \;
find . -type f -exec chmod 644 {} \;
find . -type f | wc -l
find . -type f -exec chmod u+r-wx,g+rw-x,o-rwx {} \;
find . -type f -execdir echo '{}' ';'
find . -type f -printf "%f\n"
find . -type f -print0
find . -type f -print | sed 's|^.*/S|S|'
find . -exec grep something {} +
find . -print | xargs grep something
find . -type f -print0 | xargs -0 chmod go+r
find . -type f -exec chmod ug=rw,o= {} \;
find . -type f -exec chmod u=rw,g=r,o= '{}' \;
find -type f | xargs chmod 775
find . -type f -exec chmod 775 {} +
find . -type f -exec file {} \;
find . -type f -print0 | xargs -0 file
find . -type f | xargs file
find -name '.?*' -prune -o \( -type f -print0 \)
find . -depth -path './.*' -prune -o -print
find -type f -name 'error.[0-9]*' -o -name 'access.[0-9]*' -o -name 'error_log.[0-9]*' -o -name 'access_log.[0-9]*' -o -name 'mod_jk.log.[0-9]*'
find -name '[error,access,error_log,access_log,mod_jk.log]*.[0-9]*' -type f
find . -regextype posix-egrep -regex '^.*/[a-z][^/]*$' -type f
find -type f -regex '.*\(\(error\|access\)\(_log\)?\|mod_jk\.log\)\.[0-9]+'
find . -type f -size +10k
find . -amin -1
find -mtime 0
find -daystart -mtime +0
find -mtime +0
find -mtime -1
find -mtime +1
find . -type f -mtime +7 -print0 | xargs    -0 rm
find . -name "filename_regex"|grep -v '.svn' -v '.pdv'|xargs grep -i 'your search string'
find . -type f -exec file {} \; | awk -F: '{ if ($2 ~/[Ii]mage|EPS/) print $1}'
find . -type f -exec file {} \; | awk -F: '{if ($2 ~/image/) print $1}'
find . -type f -exec file {} \; | grep -o -P '^.+: \w+ image'
find . -name '*' -exec file {} \; | grep -o -P '^.+: \w+ image'
find . -type f -ctime -3 | tail -n 5
find . -type f -ctime -3 -printf "%C@ %p\n" | sort | tail -n 5 | sed 's/[^ ]* \(.*\)/\1/'
find . -type f -perm 755 -exec chmod 644 {} \;
find -type f -printf "%s %p\n" | sort -nr | head -n 4
find -type f -printf "%s %p\n" | sort -nr | head -n 4 | awk "{print $2}"
find -type f -printf "%s %p\n" | sort -nr | head -n 4 | cut -d ' ' -f 2
find -type f -printf "%s %p\n" | sort -nr | head -n 4 | sed -r 's/[0-9]+\s//g'
find . -name .snapshot -prune -o \( -type f -mtime 0 -print0 \)
find /path/to/dir ! -perm 0644
find /path/to/dir ! -perm 644
sudo find foldername -type f -exec chmod 644 {} ";"
find images -type f
find ./subfolder ./subfolder/*/ -maxdepth 1 -type f
find media/ -type f -exec chmod 600 {} \;
find .  -mindepth 1 -type f
find path_to_dir -type f
find . -type f | xargs -I {} chmod --reference {} ../version1/{}
find . -maxdepth 1 -not -samefile /home/nez/file.txt
find . -maxdepth 1 -not -iname file.txt
find . -maxdepth 1 -not -iwholename '*Video'
find var/ -type f -exec chmod 600 {} \;
find ~/code -name '.*' -prune -o -type f -a -perm /u+x -print | sed 's@/[^/]\+$@:@' | sort | uniq | tr -d '\n' | sed 's/^/:/; s/:$//'
find -newer /etc/passwd
find . -name [ab]* -print
find / -user lal -exec chown ravi {} \;
find . -inum 211028 -exec mv {} newname.dir \;
find . -type f -not -name "*.html"
find . -exec file {} \; | grep text | cut -d: -f1
find /usr/ports/ -name Makefile\* -mindepth 3 -maxdepth 3 -exec egrep "NOPORTDOCS|NOPORTEXAMPLES" '{}' '+' | wc -l
find /usr/ports/ -name Makefile\* -exec grep -l QMAKESPEC '{}' '+' | wc -l
find /usr/ports/ -name Makefile\* -exec grep -l QTDIR '{}' '+' | wc -l
find /usr/local/games -name "*xpilot*"
find / -fstype ext3 -name zsh*
find -name '*macs'
find -L /home/peter -name *~ -exec rm '{}' +
find -L /home/peter -name *~ -exec rm '{}' \;
find -L /home/peter -name *~ -print0 |xargs -0 -r -n1000 rm
find -L /home/peter -name *~ -print0 |xargs -0 -r rm
find -L /home/peter -name *~ |xargs rm
find / -user user1
find / -type f -perm 777
find /etc -maxdepth 2 -name "*.conf" | tail
find /etc -name "*.conf" -mmin -30
find /home -user exampleuser -mtime 7 -iname ".db"
find . -maxdepth 1 -iname "*.jpg" | xargs tar -czvf jpeg.tgz
find "$dir" -maxdepth 1 -type f -iname '*.txt' | sort -n
find /home/my_dir -name '*.txt' | xargs grep -c ^.*
find . -type f -perm 644 -exec chmod 664 {} \;
find /home -type f -name '*.aac'
find `pwd` -name file.ext |xargs -l1 dirname
find ./ -name "*.txt" | rev | cut -d '/' -f1 | rev
find . -type f -name '* *'
find / -perm -u+s
find / -perm -u+s
find / -perm -u+s
find . -type f -iname '*.jpg' -print0
find -name "MyCProgram.c"
find -iname "MyCProgram.c"
find ~ -type f -mtime -2
find .
find . -name \*.php
find / -xdev -name "*.rpm"
find $dir -mtime +3
find ~ -name 'top*' -newer /etc/motd
find /usr/local -iname "*blast*"
find /usr/local -name "*blast*"
find /etc -print0 | xargs -0 file
find /etc -print0 | grep -azZ test | xargs -0 file
find /eserver6 -L
find /etc -iregex '.*xt.*'
find . -iname '*blah*' -mtime -2
find /usr/share -name '*farm*'
find . -name '*foo*'
find . -iname '*something*'
find -print0 | grep -vEzZ '(\.git|\.gitignore/)'
find . -follow -uid 0 -print
find -L /path/to/dir/* -printf "%TY%Tm%Td%TH%TM%TS|%p\n"
find \( -size +100M -fprintf /root/big.txt '%-10s %p\n' \)
find . -name .snapshot -prune -o \( \! -name *~ -print0 \)
find -mindepth 1 -maxdepth 1
find *
find . -exec $0 {} +
find /folder/path/* -mmin +120 -delete
find /tmp/test/* -daystart -mtime +0
find /tmp/test/* -daystart -mtime -1
find / -perm /u+rw,g+r,o+r
find / -perm /711
find / -size -50c
find / -size +700M
find / -size 50c
find / -group shadow
find / -user syslog
find / -perm /222
find / -perm /a+w
find / -perm /u+w,g+w,o+w
find / -perm 644
find -mindepth $i -maxdepth $i "$@"
echo "$queue" | xargs -I'{}' find {} -mindepth 1 -maxdepth 1 $*
find -mindepth 2 -maxdepth 2
find . -maxdepth 1 ! -perm  -o=r
find .*
find -regex .*sql.*
find . -regex ".*\\.rb$"
find -name "$something"
find "$STORAGEFOLDER" -name .todo  -exec dirname {} \;
find "$STORAGEFOLDER" -name .todo -printf '%h\n'
find "$STORAGEFOLDER" -name .todo -printf '%h\n'
find ./ -name Desktop
find / -user root -name FindCommandExamples.txt
find /root -name FindCommandExamples.txt
find ~/Books -name Waldo
find  / -name "apt"
find /home/username/public_html/images -name "articles.jpg"
find images -name "articles.jpg"
find . -name "articles.jpg"
find ~/Library/ -iname "com.apple.syncedpreferences.plist"
find /usr -name date
find /usr -maxdepth 4 -name document -print
find / -name document -print
find / -xdev -name document -print
find -name file -print0
find -name file -prune
find . -name file1 -print
find . -name file_name
find / -user username -group groupname -name filename
find  / -iname findcommandexamples.txt
find / -name foo
find . -name foo -type d -prune -o -name foo
find / -name foo.bar -print
find ./dir1 ./dir2 -name foo.bar -print
find . -name foo.rb
find /usr/src -name fprintf.c
find . -name game
find /usr -name java
find . -name 'javac'
find / -name "my.txt"
find ~ -name myfile
find . -name "openssl" | sed '/Permission denied/d;'
find . -name "pattern" -print
find . -iname "photo.jpg"
find -name photo.jpg
find . -name photo\?.jpg
find . -name photoA.jpg
find -iname "query"
find -iname "query"
find -name "query"
find -name "query"
find . -name test
find . -name testfile.txt
find -name "text"
find / -iname top
find / -name top
find / -name vimrc
find / -name "имя_файла"
find . -name Root | xargs cp newRoot
find -mindepth 2 -maxdepth 3 -name file
find -mindepth 4 -name file
find -maxdepth 2 -name file1
find -name file1
find . -name modules
find . -name modules \! -exec sh -c 'find -name modules' \;
sudo find / -name "orm.properties"
find /eserver6 -name "orm.properties"
find /eserver6/share/system/config/cluster -name "orm.properties"
find . -name "orm.properties"
find / \( -newer ttt -or -user wnj \) -print
find -not -name "query_to_avoid"
find \! -name "query_to_avoid"
find /home -user bob
find /some/directory -user joebob -print
find -user michel
find /var/log/crashes -name app-\*\.log -mmin -5
find . -mindepth 1 -name 'onlyme*'
find /usr/share/doc -name '[Rr][Ee][Aa][Dd][Mm][Ee]*'
find . -maxdepth 1 -name 'onlyme*'
find /home/apache -size 100c -print
find . -newer  backup.tar.gz
find / \( -newer ttt -or -user wnj	\) -print
find /file/path ! -newermt "Jul 01"
find ~ -user dave -print
find -user eric -print
find -user takuya
find / -newer ttt -user wnj -print
find /apps -group accts -print
find /dev -group audio
find /usr -group staff
find . -iregex ".*packet.*" ! -type d -size +1500c
find -maxdepth 1 -iname "*target*"
find . -name '*$VERSION*'
find . -name '*`$VERSION`*'
find /home -nouser -print
find . ! -name "*photo*"
find . -user my_user -perm -u+rwx
find /home/spenx/src -name "a1a2*txt" | xargs -n 1 dirname | xargs -I list mv list /home/spenx/dst/
find . -maxdepth 2 -name 'onlyme*'
find /usr/share/doc -name '[Rr][Ee][Aa][Dd][Mm][Ee]*'
find ~ -iname "screen*"
find ~ -iname "screen*" | more
find . -name 'test*'
find /usr -newermt "Feb 1"
find "$1" -perm -u+x -print0 | xargs chmod g+x
find $1 -perm -u+x -exec chmod g+x {} \;
find "$1" -perm -u+r -print0 | xargs chmod g+r
find $1 -perm -u+r -exec chmod g+r {} \;
find "$1" -perm -u+w -print0 | xargs chmod g+w
find $1 -perm -u+w -exec chmod g+w {} \;
find $TARGET_DIR -regextype posix-extended -regex \".*/$now.*\" -fprint $FILE_LIST
find "$dir"
find $dir -mtime -3
find $something
find /abs/path/to/directory -maxdepth 1 -name '.*invalidTemplateName.*'
find /directory_path -mtime -1 -print
find /etc -size +5M -exec ls -sh {} +
find /etc -newer /etc/motd
find /home/exampleuser/ -name "*conf" -mtime 3
find /home/user/ -cmin 10 -print
find /home/user/ -cmin 10 -print
find /usr -mmin 5
find /usr -mtime +356 -daystart
find /usr/local -iname "*blast*"
find /usr/local/games -name "*xpilot*"
find /usr/share/data -regextype posix-extended -regex ".*/20140624.*" -fprint /home/user/txt-files/data-as-of-20140624.txt
find /usr/share/doc -iname readme\*
find /usr/share/doc -name README\*
find /var/log -daystart -mtime 0
find /var/tmp -uid 1000
find A \! -path "A/a/*" -a \! -path "A/a"
find A \! -path "A/a/*"
find 'my key phrase'
find .. -exec cp -t ~/foo/bar -- {} +
find ./var/log
find /home/baumerf/public_html/ -mmin -60 -not -name \*.log
find /home/baumerf/public_html/ -mmin -60 -not -name error_log
find /home/bozo/projects -mtime 1
find /home/feeds/data
grep ! error_log | find /home/foo/public_html/ -mmin -60
find -L /myfiles
find /myfiles -size 5
find /myfiles -atime +30
find /myfiles -mtime 2
find /path -mtime +30m
find /path/to/dir -type d -exec chmod 0755 '{}' \; -o -type f -exec chmod 0644 '{}' \;
find /path/to/dir/* -printf "%TY%Tm%Td%TH%TM%TS|%p|%l\n"
find /proc -exec ls '{}' \;
find /usr/tom | egrep '*.pl| *.pm'
find /var/log
sudo find /var/www/some/subset -print0 | xargs -0 chown www-data:www-data
find _CACHE_*
find /export/home/someone -exec curl -u someone:password -vT {} ftp://somehost/tmp/
find -print
find -print0 | xargs -0
find .
find ./
find | xargs
find -print0
find |wc -l
find . | awk -F"/" '{ print $2 }'
find . -exec echo {} ";"
find . -exec echo {} ';'
find . -exec echo {} +
find . -exec echo {} \+
find . -exec echo {} \;
find . -print0 | xargs -I{} -0 echo {}
find | xargs -i sh -c "echo {} {}"
find -print | xargs -d'\n'
full_backup_dir="`find . -depth -print0`"
find . -type f -exec chmod 775 {} \;
find | sort
find -print0
find -s
find . -regex-type posix-extended -regex ".*def/incoming.*|.*456/incoming.*" -prune -o -print
find -L
find -maxdepth 3
find -iname pattern
find . -size 10M
find . -size +10M
find -E . -iregex '.*/(EA|FS)_.*'
find . -iname "{EA,FS}_*"
find . -iregex '.*/\(EA\|FS\)_.*'
find . -iregex './\(EA\|FS\)_.*'
find . -iregex './\(RT\|ED\).*' | head
find -amin 30
find . -atime -1 -print
find -amin +25 -amin -35
find . -mmin 30
find . -newer /reference/file
find | xargs
find -not -name "query_to_avoid"
find \! -name "query_to_avoid"
find . -not -path '*/\.*'
find . -newer  backup.tar.gz
find . -uid 0 -print
find -group compta
find ./ -user tom
find . -user john
find . -name "*[1k]*"
find -name "*pattern*"
find . -iname '*blah*' \( -type d -o -type f \) -mtime -2
find . -name "R*VER" -mtime +1
find . -name 'test*' -prune
find . -regex ".*/(test)[0-9][0-9]\.txt"
find . -name 'some_text_2014.08.19*'
find . -regextype sed -regex "./test[0-9]\{2\}.txt"
find . -iwholename "*ACK*1"
find . -regex filename-regex.\*\.html
find . -path './sr*sc'
find . -path './src/emacs' -prune -o -print
find . -name ”*.old” -print
find . -inum 211028 -exec mv {} newname.dir \;
find -perm -644
find . | grep "FooBar" | tr \\n \\0 | xargs -0 -I{} cp "{}" ~/foo/bar
find . | grep FooBar | xargs -I{} cp {} ~/foo/bar
find .|grep "FooBar"|xargs -I{} cp "{}" ~/foo/bar
find . -iname "*foobar*" -exec cp "{}" ~/foo/bar \;
find dir -depth
find -print0
find . -print0
find -print0 | tr "\0" ":"
find -print0
find .cache/chromium/Default/Cache/ -mindepth 1 -size +100M -delete
find "$FOLDER" -mindepth 1 | sort
find . -mindepth 2 | xargs chmod 700
find test
find whatever ... | xargs -d "\n" cp -t /var/tmp
find /tmp/test/* -daystart -mtime -0
find /tmp/test/* -mtime -1
find /usr/share/doc -iname readme\*
find . -name *.bar -maxdepth 2 -print
find . -name '*.[ch]' -exec grep $i {} | less
find . -name '*.[ch]' | xargs grep $1 | less
find /home/username/ -name "*.err"
du -a $directory | awk '{print $2}' | grep '\.in$'
find -name "*.js" -not -path "./directory/*"
find . -not \( -path ./directory -prune \) -name \*.js
find . -path ./directory -prune -o -name '*.js' -print
find /var/log -group adm -name "*.log"
find /var/log/crashes -name app-\*\.log -mmin -5 -print | head -n 1
find /storage -name "*.mp4" -o -name "*.flv" -type f | sort | head -n500
find /lib/modules -name '*.o'
find . -name "*.pdf" -print | grep -v "^\./pdfs/"
find $DIR/tmp/daily/ -name '*.tar.gz' | sort -n | tail -3
find /home -name "*.txt" -size -100k
find /home -name "*.txt" -size 100k
find /home -name "*.txt" -size +100k
find . -name '*.what_to_find' | grep -v exludeddir1 | grep -v excludeddir2
find "/cygdrive/e/MyDocs/Downloads/work/OATS Domain related/" -iname "log4j*.xml" | xargs -I % grep -ilr "CONSOLE" "%" | xargs -I % grep -H "ASYNC" %
find . -name '*my key phrase*'
find / -perm 644
find -perm 664
find ./ -perm 755
find /apps/audit -perm -7 -print | xargs chmod o‑w
find . -perm 777 -print
find -mindepth 10 -iname $TARGET
find /path -perm /011
find -inum 16187430 -exec mv {} new-test-file-name \
find "$DIR_TEMPORAL" "$DIR_DESCARGA" -maxdepth 2 -name "$nombre" -printf '%f.torrent\n'
find "$directory" -perm "$permissions"
find /tmp/ -depth -name "* *" -execdir rename " " "_" "{}" ";"
find /tmp/ -depth -name "* *" -execdir rename 's/ /_/g' "{}" \;
find $1 -name '* *'
find . -name '* *'
find . -depth -name "* *" -execdir rename "s/ /_/g" "{}" \;
find ~/Library -name '* *'
find . -uid 120 -print
find /u/bill -amin +2 -amin -6
find /path/to/look/in/ -type d -name '.texturedata' -prune
find . -not -name '*.png' -o -type f -print | xargs grep -icl "foo="
find /usr/local/fonts -user warwick
find ./ -name "foo.mp4" -exec echo {} \;
find . -name foo.mp4 -exec dirname {} \;
find . -name foo.mp4 -printf '%h\n'
find . -name foo.mp4 | sed 's|/[^/]*$||'
find ./ -name "foo.mp4" -printf "%h\n"
find . -samefile /path/to/file
find /home -xdev -samefile file1
find /tmp -type f -name ".*"
find . -type d -name ".*"
find /tmp -type f -name ".*"
find . -type f -name ".*"
find $FOLDER -name ".*"
find /tmp -type f -name ".*"
find . -type f -name ".*"
find /tmp -type f -name ".*"
find / -name httpd.conf
find ./polkadots -type f -name "image.pdf"
find ./polkadots -name 'image.pdf'
find ./polkadots -name "image.pdf" -print0
find -name 'index.*'
sort file | uniq | cut -f1 -d' ' | uniq -c | rev
find foo -type f ! -name '*Music*' -exec cp {} bar \;
find -regex '.*/modules\(/.*\|$\)' \! -regex '.*/modules/.*/modules\(/.*\|$\)' -type d -links 2
find . -maxdepth 1 -type d
grep  $USER file |nl
find . -lname /path/to/foo.txt
find -L -samefile path/to/file
find $HOME -name 'mysong.ogg'
find -user jzb
find / -type c
find / -user root -name FindCommandExamples.txt
find / -user root -name tecmint.txt
find . -name 'orm*'
find . -name "orm.*"
find /some/dir -name "*.pdf" ! -name "*_signed.pdf" -print0
find /dir/containing/unsigned -name '*.pdf' -print0
find . -type f -name "*.php"
find . \( -name "*.php" \) -exec grep -Hn "<\?php /\*\*/eval(base64_decode(.*));.*\?>" {} \; -exec sed -i '/<\?php \/\*\*\/eval(base64_decode(.*));.*\?>/d' {} \;
find . -type f -name "*.php"
find -user takuya -name '*.php' -daystart -mtime -1
find . -type f -name "*.php"
find . -type f -name tecmint.php
find . -type f -name tecmint.php
find "$topdir" -name '*.py' -printf '%h\0' | xargs -0 -I {} find {} -mindepth 1 -maxdepth 1 -name Makefile -printf '%h\n' | sort -u
find . -name '*.py' | tee output.txt | xargs grep 'something'
find /home -type f -perm /u=r
find / -perm /u=r
find ~ -name readme.txt
find . -type f -name "*.css"
find /the/path -type f -name '*.abc' -execdir rename 's/\.\/(.+)\.abc$/version1_$1.abc/' {} \;
find /var/www -type f -name "*.html"
find / -type f -name *.mp3 -size +10M -exec rm {} \;
find $dir -maxdepth 1 -type f
find .git -type f -print0 | xargs -0 sed -i 's/subdomainB\.example\.com/subdomainA.example.com/g'
find /usr/bin -type f -atime +20
find /usr/bin -type f -mtime -10
find ~ -type f -mmin -90
find * /home/www -type f
find "$dir" -mindepth 1 -type f
find $dir -maxdepth 1 -type f
find aaa/ -maxdepth 1 -type f
find /path/to/base/dir -type f
find ./ -type f -exec chmod 644 {} \;
find . -type f
find -maxdepth 1 -type f | xargs grep -F 'example'
find -type f -printf '.' | wc -c
find -type f | wc -l
find . -type f -exec echo mv -t . {} +
find -type f -print0 | xargs -r0 grep -F 'example'
find . \( -type d -regex '^.*/\.\(git\|svn\)$' -prune -false \) -o -type f -print0
find . -path ./source/script -prune -o -type f -print;
find ./ -daystart -mtime -3 -type f  ! -mtime -1 -exec ls -ld {} \;
find ./ -daystart -mtime -3 -type f  ! -mtime -1  -printf '%TY %p\n'
find ./ -daystart -mtime -3 -type f  ! -mtime -1  -printf '%Tc %p\n'
find ./ -daystart -mtime -3 -type f  ! -mtime -1  -printf '%Tm %p\n'
find -type f ! -perm -444
find . -type f ! -perm -444
find . -type f \( -exec grep -q '[[:space:]]' {} \; -o -print \)
find . –type f -mmin -10
find . -type f -name 'btree*.c'
find . -type f -name '*.DS_Store' -ls -delete
find .  -name .git -prune -o -type f -print
find /  -type f -group users
find ~ -type f -mtime 0
find ~/mail -type f | xargs grep "Linux"
find /srv/www /var/html -name "*.?htm*" -type f
find . -name Chapter1 -type f -print
find ~/Books -type f -name Waldo
find ~/Books -type f -name Waldo
find . \( ! -regex '.*/\..*' \) -type f -name "whatever"
find . -type f -name "postgis-2.0.0"
find ~/ -type f -name "postgis-2.0.0"
find /tmefndr/oravl01 -type f -newer /tmp/$$
find / -name myfile -type f -print
find / -type f -size +20000k
find / -type f -size +20000k -exec ls -lh {} \; | awk '{ print $8 ": " $5 }'
find / -mount -depth \( -type f -o -type l \) -print
find . -type f
find . -type f -print | xargs grep -i 'bin/ksh'
find / -type f -perm 0777
find ~/container  -mindepth 3 -type f  -execdir mv "{}" $(dirname "{}")/.. \;
find ~/container -mindepth 3 -type f -execdir mv "{}" ./.. \;
find ~/container  -mindepth 3 -type f -exec mv {} . \;
find ~/container  -mindepth 3 -type f -exec mv {} .. \;
find . -type f
find  /root -type f -iname "*linux*"
find . -type f -mtime 0
find . -type f -mtime +0
find . -type f -mtime +1
find . -type f -mtime +2
find . -type f -mtime +3
find . -type f -mtime +4
find . -type f -mtime +5
find . -type f -mtime +7
find . -type f –iname stat*
find . -type f -mtime $FTIME
find /path-to-directory -type f -mtime +60 -printf "%T@ %p\n" | sort
find /usr/bin -type f -size -50c
find "$somedir" -type f -exec echo Found unexpected file {} \;
find ${DIR} -type f -regex ".*\.${TYPES_RE}"
find $DIR -type f -iname "*.$TYPE"
find $DIR/tmp/daily/ -type f -printf "%p\n" | sort -rn | head -n 2 | xargs -I{} cp {} $DIR/tmp/weekly/
find "$DIRECTORY_TO_PROCESS" -type f -iregex ".*\.$FILES_TO_PROCES" ! -name "$find_excludes" -print0
FILES=$(find $FILES_PATH -type f -name "*")
find ${FOLDER} -type f ! -name \".*\" -mtime -${RETENTION} | egrep -vf ${SKIP_FILE}
find $SOURCE -type f -mtime +$KEEP | sed ‘s#.*/##'
find "$d/" -type f -print0 | xargs -0 chmod 777
find $dir -type f
find $dir -type f -size +"$size"M -printf '%s %p\n' | sort -rn
find $dir -type f -name $1 -exec sed $num'q;d' {} \;
find "${S}" -type f
find ${path} -P -type f
find /directory_path -type f -mtime -1 -print
find /home/john -name "landof*" -type f -print
find /home/john/scripts -type f -not -name "*.ksh" -print
find /usr/bin -type f -size -50c
find ./Desktop -type f
sed -i '' -e 's/subdomainA/subdomainB/g' $(find /home/www/ -type f)
find /home/www -type f -print0 | xargs -0 sed -i 's/subdomainA\.example\.com/subdomainB.example.com/g'
find /home/www/ -type f -exec sed -i 's/subdomainA\.example\.com/subdomainB.example.com/g' {} +
find . /home/admin/public_html/ -type f -exec chmod 644 {} \;
find /home/user/demo -type f -print
find /root -type f -iname "*linux*"
find /somepath -type f -iregex ".*\.(pdf\|tif\|tiff\|png\|jpg\|jpeg\|bmp\|pcx\|dcx)" ! -name "*_ocr.pdf" -print0
find . -depth -type f -print
find . -type f
find . \( ! -regex '.*/\..*' \) -type f -print0 | xargs -0 sed -i 's/subdomainA.example.com/subdomainB.example.com/g'
find . -maxdepth 1 -type f -print0 | xargs -0 sed -i 's/toreplace/replaced/g'
find . -maxdepth 1 -type f -perm -uga=x
find . -type f -exec sed -i "s/1\.2\.3\.4/5.6.7.8/g" {} \
find . -type d -path '*/\.*' -prune -o -not -name '.*' -type f -name '*some text*' -print
find . \( -not -path './dir1/*' -and -not -path './dir2/*' -or -path './dir1/subdir1/*' \) -type f
find . -not -path '*/\.*' -type f -name '*some text*'
find . -type f -name \* | grep "tgt/etc/*"
find . -type f -atime $FTIME
find . -type f \( -name "*cache" -o -name "*xml" -o -name "*html" \)
find . \( -path './dir1/*' -and -not -path './dir1/subdir1*' -or -path './dir2' \) -prune -or -type f -print
find . -type f -exec sed -i ‘s/.*abc.*/#&/’ {} \;
find test -type f
find ~/$folder -name "*@*" -type f
find ~/$folder -name "*@*" -type f -print0
find . -type f -iname '*'"$*"'*' -ls
find . -type f -not -name "*.html"
find . -type f|grep -i "\.jpg$" |sort| tee file_list.txt
find ./ -type f \( -name '*.r*' -o -name '*.c*' \) -print
find . -type f -name "*.txt" ! -name README.txt -print
find / -iname "*.what_to_find" -type f -exec mv {} /new_directory \;
find /data -type f -perm 400 -print
find . -type f -perm 755 -exec chmod 644 {} \;
find . -type f -iname '*'"${1:-}"'*' -exec ${2:-file} {} \;
find "$fileloc" -type f -prune -name "$filename" -print
find /home/user/demo -type f -perm 777 -print
find /home/user/demo -type f -perm 777 -print -exec chmod 755 {} \;
find -name "* *" -type f | rename 's/ /_/g'
find . -perm -g=r -type f -exec ls -l {} \;
find . -not -path '*/\.*' -type f \( ! -iname ".*" \)
find . -name "sample*" | xargs -i echo program {}-out {}
find . -name "sample*_1.txt"
find . -name "sample*_1.txt" | sed -n 's/_1\..*$//;h;s/$/_out/p;g;s/$/_1.txt/p;g;s/$/_2.txt/p' | xargs -L 3 echo program
find . -type s
cat "$FILE" | grep "^${KEY}${DELIMITER}" | cut -f2- -d"$DELIMITER"
find -maxdepth 1 -type d ! -name ".*"
find /usr/bin -name '*vim*' -type l
find /usr/ -lname *javaplugin*
find /usr/bin -name '*vim*' -type l
find -type l
find . -type l -ls
find "/proc/$pid/fd" -ignore_readdir_race -lname "$save_path/sess_\*" -exec touch -c {}
find /some/directory -type l -print
find /some/directory -type l -print
find . -type f -links 1 -print
find –L –xtype l
find /myfiles -type l
find /myfiles -type l
find . -type l
find ./ -type l
find /var/log -name "syslog" -type d
find . -type l -exec readlink -f '{}' \; | grep -v "^`readlink -f ${PWD}`"
find . -name test.txt
sudo find . -name test1.h
sudo find . -name test2.h
find ~/ -name '*.txt'
find ~/Programming -path '*/src/*.c'
find . -name "*.pl"
find . -maxdepth 1 -name '*.txt' -mtime +2
find . -name '*.c' | xargs grep 'stdlib.h'
find . -name ‘*.c’ | xargs egrep stdlib.h
find ./music -name "*.mp3" -print0 | xargs -0 ls
find ./music -name "*.mp3" -print0 | xargs -0 ls
find . -perm 0644 | head
find / -perm 2644
find / -perm 2644
find . -perm /g+s
find . -perm /u=s
find / -perm 0551
find / -perm 0551
find / -perm 1551
find / -perm 1551
find / -user root -name FindCommandExamples.txt
find / -atime 50
find / -mtime 50
find / -mtime +50 -mtime -100 | head
find / -size +50M -size -100M
find / \( -name '*.txt' -o -name '*.doc' -o -size +5M \)
find / -atime 50
find / -amin -60
find / -cmin -60
find / -size +50M -size -100M
find / -mtime 50
find / -mmin -60
find / -mtime +50 –mtime -100
find * -type f -print -o -type d -prune
find . -perm 0644 | head
find . -type f -perm 0777 -print
find . -type f ! -perm 777 | head
find . -maxdepth 1 -iname "*linkin park*"
find / -name vimrc
find / -amin -60
find / -cmin -60
find / -mmin -60
find /tmp/test/* -mtime -0
find -maxdepth 1 -not -iname "MyCProgram.c"
find . -name "*.java"
find . -mtime 1
find . -mtime +1
find . -mtime -1
find /etc -size +100k
find /home -name tecmint.txt
find /tmp  | head
find / -atime 50
find / -amin -60
find / -amin -60
find / -cmin -60
find / -cmin -60 | head
find / -size +50M -size -100M
find / -size +50M -size -100M
find / -mtime 50
find / -mmin -60
find / -mmin -60
find / -mtime +50 –mtime -100
find / -atime 50
find / -mtime 50
find / -mtime +50 -mtime -100
find  / -iname findcommandexamples.txt
find /root -name FindCommandExamples.txt
find . -name tecmint.txt
find /home -iname tecmint.txt
find . -name tecmint.txt
find . -type f -perm 0777 -print
find . -type f -perm 0777 -print
find . -type f -perm 0777 -print
find / -type f ! -perm 777
find / -type f ! -perm 777
find . -type f ! -perm 777 | head
find "/path/to/files" -mmin +120
find ./ -newermt 2014-08-25 ! -newermt 2014-08-26 -print
find / \! \( -newer ttt -user wnj \) -print
find /usr/local -mtime -1
find /var/adm -mtime +3 -print
find ~ ! -user ${USER}
find /etc -maxdepth 1 -name "*.conf" | tail
find "$DIR" -type f -mtime +15 -exec rm {} \;
find $DIR -type f -mtime +450 -exec rm {} \;
find "$DIR" -type f \! -newer "$a" \! -samefile "$a" -delete
find "$DIR" -type f \! -newer "$a" \! -samefile "$a" -exec rm {} +
find /your/dir -type f -size +5M -exec du -h '{}' + | sort -hr
find . -type f -mtime +31 -print0 | xargs -0 -r rm -f
find dir1 -mindepth N -type f
find /home/backups -type f \( -name \*.tgz -o -name \*.gz \) -print0 | xargs -0 ls -t | tail -1 | xargs rm
find temps/ -name "thumb.png"
find . -maxdepth 1 -name '*Music*' -prune -o -print0 | xargs -0 -i cp {} dest/
find . -name '*.xml'
find -name \*.xml -print0 | xargs -0 -n 1 -P 3 bzip2
find ~ -name 'xx*' -and -not -name 'xxx'
find -name \*.jsp | sed 's/^/http:\/\/127.0.0.1/server/g' | xargs -n 1 wget
find . -name "*.txt" -print
find . -name "*.txt" -print | less
find . -inum 968746 -exec rm -i {} \;
find . -type f -exec sed '1s/^\xEF\xBB\xBF//' -i.bak {} \; -exec rm {}.bak \;
find . -name "*.pl" | xargs tar -zcf pl.tar.gz
find . -name \*.log -print0 | xargs -I{} -0 cp -v {} /tmp/log-files
find . -depth -name '*.zip' -exec rm {} \;
find ~/ -name 'core*' -exec rm {} \
rm `du * | awk '$1 == "0" {print $2}'`
find /home -xdev -samefile file1 -exec rm {} +
find /home -xdev -samefile file1 -print0 | xargs -0 rm
find /home -xdev -samefile file1 | xargs rm
find ./ -inum 1316256 -delete
find | head
find . -type f -ls
find . -name "*.pdf" -print | grep -v "^\./pdfs/"
find . -perm g=r -type f -exec ls -l {} \;
find . -name "*.pdf" -print
find . -print
find .
find . -print
find / -name "*.core" -print -exec rm {} \;
find / -name "*.core" | xargs rm
find . -type f -name "*.mp3" -exec rm -f {} \;
find . -type f -name "*.txt" -exec rm -f {} \;
find . -type f -name "*.mp3" -exec rm -f {} \;
find . -type f -name "*.txt" -exec rm -f {} \;
find . -type f -name "*.mp3" -exec rm -f {} \;
find . -type f -name "*.mp3" -exec rm -f {} \;
find . -type f -name "*.txt" -exec rm -f {} \;
find /home -name .rhosts -print0 | xargs -0 rm
find . -inum 782263 -exec rm -i {} \;
find /usr/* -size 0c -exec rm {} \;
find . -size 2000k
find . -size -500k
find / -size +900M
find /home/tecmint/Downloads/ -type f -printf "%s %p\n" | sort -rn | head -n 5
find . -name '*.csv.gz' -exec gzip -d {} \;
find . -name '*.csv.gz' -print0 | xargs -0 -n1 gzip -d
find . -name *disc*
find . -atime +7 -size +20480 -print
find . -atime +7 -o -size +20480 -print
find . -type f -name ".*" -newer .cshrc -print
du -hs /path/to/directory
find . -depth -name "blabla*" -type f | xargs rm -f
find / -type l -print0 | xargs -0 file | grep broken
find / -type l -print0 | xargs -0 file | grep broken
find ./ -follow -lname "*"
find -L -type l
find . -type l -xtype l
find . -type f -exec ls -s {} \; |sort -n -r |head
find . -type f -exec ls -s {} \; sort -n |head -5
find . -exec echo ' List of files & Direcoty'   {} \;
find . -type f -and -iname "*.deb"
find . -iname '*blah*' \( -type d -o -type f \) -mtime -2
find /proc -type d | egrep -v '/proc/[0-9]*($|/)' | less
find /path -type d -printf "%f\n" | awk 'length==33'
find . -maxdepth 1 -type d -print | xargs  -I "^" echo Directory: "^"
find -type d ! -perm -111
find . -depth -type d -mtime 0 -exec mv -t /path/to/target-dir {} +
find . -type d -mtime -0 -exec mv -t /path/to/target-dir {} +
find . -type d -mtime -0 -print0 | xargs -0 mv -t /path/to/target-dir
find . -type d -mtime 0 -exec mv {} /path/to/target-dir \;
find . -type d -name "?????????????????????????????????"
find . -mtime -7 -type d
find . -mtime -7 -type d
find /usr/ports/ -name work -type d -print -exec rm -rf {} \;
find . -type d -name build
find /usr -name doc -type d
find /usr \( -name doc -and -type d \)
find / -user news -type d -perm 775 -print
find /TBD -mtime +1 -type d
find $workspace_ts -mindepth 1 -maxdepth 1 -type d -mtime -30
find $workspace_ts -mindepth 1 -maxdepth 1 -type d -mtime +30 -print
find /home/user/workspace -mindepth 1 -maxdepth 1 -type d -mtime +30 -execdir echo "It seems that {} wasn't modified during last 30 days" ';'
find /home/user/workspace -mindepth 1 -maxdepth 1 -type d -mtime +30 -printf "\t- It seems that %p wasn't modified during last 30 day\n"
find . -type d -perm 755 -exec chmod 700 {} \;
find $dir -maxdepth 1 -type d -user $username -perm -100
find /home -type d -perm 777 -print -exec chmod 755 {} \;
find /some/dir/ -maxdepth 0 -empty
find your/dir -prune -empty
find your/dir -prune -empty -type d
du -a
du --max-depth=0 ./directory
find . -empty
find test -empty
find test -empty
find test -empty
find /dir -type f -size 0 -print
find wordpress -maxdepth 1 -name '*js'
find wordpress -name '*js'
find 0001 -type d | sed 's/^0001/0002/g' | xargs mkdir
find /home -user joe
find /usr -name *stat
find /var/spool -mtime +60
find /var/spool -mtime +60
find /home -user joe
find -type f -iname '*.un~'
find -type f -iname '*.un~'
find . -perm -100 -print
find / -name Chapter1 -type f -print
find /home -name foo.bar -type f -exec rm -f "{}" ';'
find /etc -name hosts
find /usr/local -name "*blast*"
find /usr/local -iname "*blast*"
du -b FILE
find -maxdepth 2 -name file1
find . -atime -1 -print
find /tmp/ -depth -name "* *" -execdir rename 's/ /_/g' "{}" \;
find . -mtime 1
find . -mtime -7
find . -mtime -7
find . -newer CompareFile -print
find . -user xuser1 -exec chown -R user2 {} \;
find /usr/src -name CVS -prune -o -mindepth 7 -print
find /usr/src -name CVS -prune -o -depth +6 -print
find . -user daniel
find . -gid 1003
find . -name RAID -prune -o -print
find . -inum 968746 -exec ls -l {} \;
find . -inum 968746 -print
find /path/to/search -user owner
find ~ -size +20M
find -type type_descriptor
find . -mtime -1 -type f
find . -iname '*blah*' -mtime -2
find . -type f -execdir /usr/bin/grep -iH '#!/bin/ksh' {} \; | tee /tmp/allfiles
find . -type f -print | xargs /usr/bin/grep -il 'bin/ksh' | tee /tmp/allfiles
find / -newerct '1 minute ago' -print
find -name '*macs'
find .  -path '*/*config'
find .  -path '*f'
find "${DIR_TO_CLEAN?}" -type f -mtime +${DAYS_TO_SAVE?} -print0
find /var/tmp/stuff -mtime +90 -print
find . -name not\* | tail -1 | xargs rm
find / -perm /g+w,o+w
find / -perm /g=w,o=w
find / -perm -u+rw,g+r,o+r
find dir -name '?????????????????????????????????'
find /home/user/ -cmin 10 -print
find /travelphotos -type f -size +200k -not -iname "*2015*"
find /var/log/ -mmin +60
find /var/log/ -mmin -60 -mmin +10
find /var/log/ -mtime +7 -mtime -8
find . -size 2000k -print
find . -size -500k -print
find . -newer /bin/sh
find . -name f* -print
find . -not \( -name .svn -prune -o -name .git -prune -o -name CVS -prune \) -type f -print0 | xargs -0 file -n | grep -v binary | cut -d ":" -f1
find -type f ! -perm -444
find . \( -size +700k -and -size -1000k \)
find . -name some_pattern -print0 | xargs -0 -i mv {} target_location
find . -links 1
find ./ -name "*sub*"
find . -amin -60
find . -size +5000k -type f
find . -perm 766
find . -mmin -60
find . -regextype posix-egrep -regex '.\*c([3-6][0-9]|70).\*'
find . -regextype posix-egrep -regex "./c(([4-6][0-9])|70)_data.txt"
find -name 'file*' -size 0 -delete
find . -name 'file*' -size 0 -print0 | xargs -0 rm
find -regex "^.*~$\|^.*#$"
find ./ | grep "sub"
find .  -perm 775
find . -size 24000c
find . -size +24000c
find . -size -24000c
find . -cmin -60
find . -name file* -maxdepth 1 -exec rm {} \;
find esofthub esoft -name "*test*" -type f -ls
find /var/www/ -type f -name "*" -size +100M -exec du -h '{}' \;|grep -v /download/
find . -size +50k
find / -name .ssh* -print | tee -a ssh-stuff
find / -perm 644
find $1 -name "$2" -exec grep -Hn "$3" {} \;
find $1 -name "$2" | grep -v '/proc' | xargs grep -Hn "$3" {} \;
find $1 -path /proc -prune -o -name "$2" -print -exec grep -Hn "$3" {} \;
find . | xargs grep regexp
find . -path "./sr*sc"
find . -newermt "5 days"
find . -mmin +5 -mmin -10
find . -mtime -7 -type f
find . -mmin -5
find . -newer poop
find . -mtime 0
find ./ -name "blabla" -exec wc -l {} ;
find . -iname "needle"
find /etc -name ppp.conf
find /tmp -depth -name core -type f -delete
find /tmp -name core -type f -print0 | xargs -0 /bin/rm -f
find /path/to/folder -name fileName.txt -not -path "*/ignored_directory/*"
find /tmp -name core -type f -print0 | xargs -0 /bin/rm -f
find /tmp -name core -type f -print | xargs /bin/rm -f
find / -user root -name tecmint.txt
find . -newer tmpfile
find ~/src -newer main.css
find ./ -newer start.txt -and ! -newer end.txt
find Folder1 \( ! -name 'Image*-70x70*' -a ! -name 'Image*-100x100*' \) | xargs -i% cp -p % Folder2
find Folder1 -type f -regextype posix-extended \( ! -regex '.+\-[0-9]{2,4}x[0-9]{2,4}\.jpg' \) -print0 |  xargs -0 cp -p --target-directory=Folder2
find / -atime -1 -amin +60
find / -type f -size +50M -size -100M
find / -ctime -50
find / -mmin +90
find / -type f -size +20M -exec ls -lh {} \; | awk '{ print $NF ": " $5 }'
find / -name "[Xx]*"
find / -nogroup
find / -nouser
find / -nogroup -print
find / -nouser -print
find / -group shadow
find . -name "pattern" -print
find . -perm g=r -type f -exec ls -l {} \;
find . -name "*.ext"
find . -size -40 -xdev -print
find . -name "file*"
find . -iname "file*"
find . -maxdepth 1 -size 0c -exec rm {} \;
find . -size 0 -exec rm {} \;
find -size 100k
find -empty -type -f
find -nouser
find / -nouser -o  -nogroup
find / -type f ! -perm 644
find . -maxdepth 1 -name \*.gz -print0 | xargs -0 zcat | awk -F, '$1 ~ /F$/'
find / -mtime 1
find /etc/ -mtime -30 | xargs -0 cp /a/path
find -type f -perm /110
find / -atime -1
find . -type f -mtime 7 | xargs tar -cvf `date '+%d%m%Y'_archive.tar`
find / -mmin -1
find . -type f -mtime -7 | xargs tar -cvf `date '+%d%m%Y'_archive.tar`
find . -type f -mtime +7 | xargs tar -cvf `date '+%d%m%Y'_archive.tar`
find . -type f -mtime +7 -mtime -14 | xargs tar -cvf `date '+%d%m%Y'_archive.tar`
find . -type f -mtime +7 -mtime -14 | xargs tar -cvf `date ‘+%d%m%Y’_archive.tar`
find /travelphotos -type f -size +200k -not -iname "*2015*"
find /etc/apache-perl -newer /etc/apache-perl/httpd.conf
find /some/path -type f ! -perm -111 -ls
find /some/path -type f ! -perm -100 -ls
find /tmp -size +10k -size -20k
find /usr -newer /usr/FirstFile -print
find /usr ! -newer /FirstFile -print
find [directory] -name "pattern_to_exclude" -prune -o -name "another_pattern_to_exclude" -prune -o -name "pattern_to_INCLUDE" -print0 | xargs -0 -I FILENAME grep -IR "pattern" FILENAME
find . -type f -newermt "$date_time"
find . -type f -not -newermt "$date_time"
find . -type f -exec grep -iH '/bin/ksh' {} \;
find . -type f -print | xargs    grep -il 'bin/ksh'
find -x . -type f -print0
find -iname "MyCProgram.c"
find ${userdir}/${i}/incoming -mtime +2 -type f -ls
find ${userdir}/${i}/incoming -mtime +2 -type f -exec rm {} \;
find . -cmin -60
find ./ -mmin +1
find ./ -daystart -mtime -10 -and -mtime +1
find . -iname 'MyFile*'
find ./ -type f -name "$2" -exec sed -i "s/$3/$4/g" {} \;
find . |xargs grep search string | sed 's/search string/new string/g'
find .  -path '*/*config'
find .  -path '*f'
find / -type f -perm -002
find / -type f -perm -002 -printf '%p has world write permissions\n'
echo $(find / -type f -perm -002) has world write permissions
find / -type f -perm -002 -print0
find /tmp -type f -perm -002 | sed '1s/^/Found world write permissions:\n/'
find /tmp -type f -perm -002 | awk -- '1{print "Found world write permissions:";print};END{if(NR==0)print "No world writable found."}'
find / -type f -perm 0777 -print -exec chmod 755 {} \;
find /  \( -perm -2000 -o -perm -4000 \) -ls
find . -name \*\\?\*
find /etc -name "*.conf"
find . -perm -20 -exec chmod g-w {} ;
find . -perm -20 -print | xargs chmod g-w
find / -inum 199053
find . -name aaa.txt
find ${userdir}/${i}/incoming -mtime +2 -type f -size +200557600c -ls
find ${userdir}/${i}/incoming -mtime +2 -type f -size +200557600c -exec rm {} \;
find /etc -name '*.conf'
find -name "*test*" -depth
find ~ -size +10M
find / -newer myfile
find / -ctime +3
find / -mmin -1
find / -atime -1
find / -mtime 1
find / -perm -644
find /etc -type f -ctime -1
find . -mtime -1
find -name TEST_3
find . -name aaa.txt
find . -name "articles.jpg" -exec chmod 644 {} \;
find / -fstype ext2 -name document -print
find / /usr -xdev -name document -print
find / -path /usr/lib/important/*/file.txt
find / -user tutonics -name "file.txt"
find / -name filename -print
find -x / -name foo
find . -name foo -type d -prune -o -name foo -print
find / -name foo.bar -print -xdev
find ./dir1 ./dir2 -name foo.bar -print
find / -name photo.jpg
find /usr /bin /sbin /opt -name sar
find `ls -d /[ubso]*` -name sar
find ./ -name "somename.txt"
find ./ -iname blah
find ./ -name blah
find -name "<filetype>" -atime -5
find /dev/shm /tmp -type f -ctime +14
find /usr/local -size +10000k
find -newer foo.txt
find / -user syslog
find /tmp -user ian
find /path ! -perm /020
find /path ! -perm /g+w
find /path ! -perm /022
find /path ! -perm -022
find /path -nouser -or -nogroup
find /tmp/test/* -daystart -mtime +1
find /var/www -group root -o -nogroup -print0 | xargs -0 chown :apache
find /var/www ! -user apache -print0 | xargs -0
find /dir -newer yesterday.ref -a \! -newer today.ref -print
find /usr -newer /tmp/stamp$$
find /tmp -size -100c
find /users/tom -name "*.pl" -name "*.pm"
find . | xargs -n 1 echo
find . -print0 | xargs    -0 echo
find -print0
find . -path ./src/emacs -prune -o -print
find . -regextype posix-egrep -regex ".+\.(c|cpp|h)$"
find . -mtime 1
find -ipath './projects/insanewebproject'
find -ipath './projects/insanewebproject'| head -n1
find -ipath 'projects/insanewebproject'
find . | grep -qi /path/to/something[^/]*$
find -maxdepth 0
find -prune
find /path -perm 777
find /path -perm ugo+rwx
find -x /var -inum 212042
find /path -perm -022
find /path -perm -g+w,o+w
find /path -perm -go+w
find /path -perm /g+w,o+w
find / -type l -lname '/mnt/oldname*'
find / -name grub.conf
find lpi104-6 -samefile lpi104-6/file1
find . -type d -exec basename {} \; | wc -l
find . -type f -exec basename {} \; | wc -l
find ~/Desktop -name “*.jpg” -o -name “*.gif” -o -name “*.png” -print0 | xargs -0 mv –target-directory ~/Pictures
find .  -maxdepth 2 -name '*.tmp'
find . -type f -printf "%s\t%p\n" | sort -n | tail -1
find . -lname \*foo.txt
find / -lname foo.txt
find -L / -samefile path/to/foo.txt
find ~/Movies/ -size +1024M
find $HOME -name 'mysong.ogg'
find /home/www/ ! -executable
find . -maxdepth 1 -name "$a" -print -quit
find . -type d
find /etc -size +100k -size -150k
find -type f -iname "*.txt" -exec ls -lrt {} \;|awk -F' ' '{print $1 $2  $9}'
find -type f -iname "*.txt" -exec ls -lrt {} \;|awk -F' ' '{print $1  $9}'
find / -user vivek -name "*.sh"
find / -user vivek
find /home -xdev -samefile file1
find / -path /proc -prune -o -user account -ls
find . -name '*~' | xargs rm
find . -name '*.py' | xargs grep 'import'
find . -name '*.py' | xargs wc -l
find -type d -empty
find . -type d -empty
find b -cmin -5
find /path -type f -name "*txt" -printf "cp '%p' '/tmp/test_%f'\n" | bash
find . -type f -name "*.class" -exec rm -vf {} \;
find xargstest/ -name 'file??' | sort
find . -name \*\:\*
find . -name "foo*"
find . -name "*foo"
find . -type f -name "*.txt" ! -name README.txt -print
find b -type f -cmin -5
find b -type f -cmin -5 -exec cp '{}' c \;
find -name '.?*' -prune -o \( -type f -print0 \)
find . -type f | grep -P "\.dll$|\.exe$"
find . -type f | grep -vP "\.dll$|\.exe$"
find . -type f -a -name '*.*'
find -type f -print0
find . -type f -print
find "$1" -path "*/.*" -prune -o \( -type f -print0 \)
find . -name .git  -prune -o -name file  -print
find . -path ./.git  -prune -o -name file  -print
find . -name "*zip" -type f | xargs ls -ltr | tail -1
find . -type f -print0 | xargs -0 ls -ltr | tail -n 1
find . -type f -print0|xargs -0 ls -drt|tail -n 1
find . -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" "
find . -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" " | sed 's/.*/"&"/' | xargs ls -l
find . -type f -printf '%TY-%Tm-%Td %TH:%TM: %Tz %p\n'| sort -n | tail -n1
find . -type f | sed 's/.*/"&"/' | xargs ls -E | awk '{ print $6," ",$7 }' | sort | tail -1
find . -type f | sed 's/.*/"&"/' | xargs ls -E | awk '{ print $6," ",$7," ",$9 }' | sort | tail -1
find . -type f | xargs ls -ltr | tail -n 1
find -maxdepth 1 -type f -perm /222
find -maxdepth 1 -type f -perm /200
find . -type f -name "FindCommandExamples.txt" -exec rm -f {} \;
find -L . -type f -perm -a=x
find -L . -type f \( -perm -u=x -o -perm -g=x -o -perm -o=x \)
find -L . -type f -perm -u=x,g=x  \! -perm -o=x
find . -type f -perm -u=x
find . -type f -size +500M
find . -name "*oraenv*" -type f -exec file {} \;
find . -mtime -7 -type f
find /etc -type f -mmin -10
find /dir/to/search/ -type f -name 'expression -and expression' -print
find /dir/to/search/ -type f -name 'expression -or expression' -print
find /dir/to/search/ -type f -name 'regex' -print
find /usr /home -name findme.txt -type f -print
find /tmp -name core -type f -print | xargs /bin/rm -f
find . -perm -g=r -type f -exec ls -l {} \;
find . -type f -size +500M
find . -type f -size +2G
find / -perm +6000 -type f
find /somefolder -type f '(' "${ARGS[@]}" ')'
find / -maxdepth 1 -xdev -type f  -exec grep -li stringtofind '{}' \;
find / -maxdepth 1 -xdev -type f -exec grep -i "stringtofind" -l {} \; -exec sed -i '/./d' {} \;
find /path -type f -name "???-???_[a-zA-Z]*_[0-9]*_*.???"
find . -perm 644 -type f -exec ls -l {} \;
find . -type f -name "*.JPG"
find -perm -111 -type f
find . -not -path '*/\.*' -type f -name '*some text*'
find . -type d -path '*/\.*' -prune -o -not -name '.*' -type f -name '*some text*' -print
find /apps/ -user root -type f -amin -2 -name *.rb
find ./ -user root
find . -type f -printf "%s\t%p\n" | sort -n |head -1
find . -type f -print | xargs grep -ni "STRING"
cat /var/run/dmesg.boot | grep Features
find . -type f -name "*.php" -exec grep --with-filename "eval(\|exec(\|base64_decode(" {} \;
find  /usr/lib/ /usr/lib64/ -lname "*libstdc++*"
find /usr/sbin /usr/bin -lname "*/systemctl"
find /etc -type l
find lpi104-6 research/lpi104-6 -lname "*file1"
find /etc -type l
find . -type l | xargs ls -ld
find test -type l -exec cp {} {}.tmp$$ \; -exec mv {}.tmp$$ {} \;
find . –name "*.txt" –mtime 5
find . -type f | xargs grep "text"
find . -name .snapshot -prune -o -name '*.foo' -print
awk -F, 'NR==1 {gsub(/"/,"",$3);print $3}' "$(dirname $(readlink -f $(which erl)))/../releases/RELEASES"
find /var  -type f -exec grep "param1" {} \; -print
find /var -type f  | xargs grep "param1"
find .  -name "*.groovy" -not -path "./target/*" -print
find /home/tecmint/Downloads/ -type f -exec du -Sh {} + | sort -rh | head -n 5
find -type f -exec du -Sh {} + | sort -rh | head -n 5
find . -name “core” -exec rm -f {} \;
find . -type d
find -mindepth 3 -maxdepth 3 -type d -name "*New Parts*" -exec ln -s -t /cygdrive/c/Views {} \;
find -mindepth 3 -maxdepth 3 -type d | grep "New Parts" | tr '\012' '\000' | xargs -0 ln -s -t /cygdrive/c/Views
find -type d -printf '%T+ %p\n' | sort | head -1
find -empty
find . -inum $inum -exec rm {} \;
find . -inum 1316256
find . -inum 211028 -exec mv {} newname.dir \;
find . -name '*.ISOLATE.*.txt' -maxdepth 1 -print0 | xargs -0 -IFILE mv FILE ./ISOLATE
find -name '*.ISOLATE.quantifier.txt' -maxdepth 1 -exec mv {} ISOLATE/ +
find . -name '*.JUKEBOX.*.txt' -maxdepth 1 -print0 | xargs -0 -IFILE mv FILE ./JUKEBOX
find . -type f -perm 644 -exec chmod 664 {} \;
find ... -print -quit
find . -name something -print -quit
/usr/bin/find $DIR -maxdepth 1 -ipath $TMP_DIR -print -quit
find . ... -print -quit
find . -type d -print0 | xargs -0 du | sort -n | tail -10 | cut -f2 | xargs -I{} du -sh {}
find . -type f -print0 | xargs -0 du | sort -n | tail -10 | cut -f2 | xargs -I{} du -sh {}
find /home/tecmint/Downloads/ -type f -exec du -Sh {} + | sort -rh | head -n 5
find /home/tecmint/Downloads/ -type f -printf "%s %p\n" | sort -rn | head -n 5
find . -name '*.gz' -print | xargs gzip -l | awk '{ print $2, $4 ;}'  | grep -v '(totals)$' | sort -n | tail -1
find . -name '*.gz' | xargs gzip -l | tail -n +2 | head -n -1 | sort -k 2 | tail -n 1 | awk '{print $NF}'
find . -type f -printf '%TY-%Tm-%Td %TT   %p\n' | sort
find -maxdepth 2 -name passwd
find -maxdepth 2 -name passwd
find / -maxdepth 3 -name passwd
find / -maxdepth 3 -name passwd
find -mindepth 3 -maxdepth 5 -name passwd
find -mindepth 3 -maxdepth 5 -name passw
ps -A|grep mysql
find . \( -name "*.php" -o -name "*.html" \) -print0 | xargs -0 grep -Hin "joomla"
find . -type f -exec ls -al {} \; | sort -nr -k5 | head -n 25
find . -type f -exec ls -s {} \; | sort -n -r | head -5
find . -type f -exec ls -s {} \; | sort -n -r | head -5
find . -type f -exec ls -s {} \; | sort -n  | head -5
find . -type f -iname '*.jpg' -print0 | du -c --files0-from=-
find . -type f -iname '*.jpg' -print0 | xargs -r0 du -a| awk '{sum+=$1} END {print sum}'
find ./photos/john_doe -type f -name '*.jpg' -exec du -ch {} + | grep total$
find /bin -type f -follow | xargs    ls -al | awk ' NF==9 { print $3 }'|sort -u
find /path/to/search -daystart -ctime -1
find . -user daniel -type f -name *.jpg
find . -user daniel -type f -name *.jpg ! -name autumn*
who | cut -d ' ' -f 1 | grep -e '^ab' -e '1$'
who | grep -e '^ab' -e '1$'
find . -name '*.rb' -exec grep -H jump {} \;
find . -type f -name 'abc*' ! -name '*.py'
find . -writable
find -type f -maxdepth 1 -writable
find .  -maxdepth 1 -type f -writable
find . -type f -writable | grep -v sites/default/files
find /tmp/ -ctime -1 -name x*
find /tmp/ -ctime -1 -name "x*" -exec mv '{}' ~/play/
find /tmp/ -ctime -1 -name "x*" | xargs -I '{}' mv '{}' ~/play/
find /tmp/ -ctime -1 -name 'x*' -print0 | xargs -r0 mv -t ~/play/
echo $b|grep -q $a
ifconfig en0 | grep inet | grep -v inet6 | awk '{print $2}'
ifconfig eth0 | grep 'inet addr:' | awk '{print $2}' | awk -F ':' '{print $2}'
ifconfig | grep 192.168.111 | awk '{print $2}'
ifconfig | grep -v '127.0.0.1' | sed -n 's/.*inet addr:\([0-9.]\+\)\s.*/\1/p'
ifconfig | sed -n 's/.*inet addr:\([0-9.]\+\)\s.*/\1/p'
ps -A -o pid | xargs -I pid readlink "/proc/pid/exe" | xargs -I file dirname "file"
find -type f -exec grep -Hn "texthere" {} +
find $LOCATION -print -exec shred $TIMES -u '{}' \;
find /path -type f -exec ls -l \{\} \;
find . -type f -ctime -$2 -name "mylog*.log" | xargs bzip2
dirname `find / -name ssh | grep bin`
find / -name ssh|grep bin|xargs dirname
w | awk '{print $1}'
pstree | grep php
x=$(grep "$(dirname "$path")" file)
find /myDir -name 'log*' -and -not -name '*.bz2' -ctime +7 -exec bzip2 -zv {} \;
find /myDir -name "log*" -ctime +7 -exec bzip2 -zv {} \;
w | sed '1,2d' | cut -f1 -d' ' | sort | uniq -c
w | grep ssh
find -type f -name 'header.php' | xargs -n 1 dirname | xargs -n 1 cp -f topscripts.php
cd $(find . -name Subscription.java | xargs dirname)
cd `find . -name Subscription.java | xargs dirname`
diff -qr directory directory.original | cut -d' ' -f2 | xargs dirname | uniq
directories=$(diff -qr directory directory.original | cut -d' ' -f2 | xargs dirname | uniq)
env | grep DUALCASE
grep -r "string to be searched"  /path/to/dir
find /home/folder1 /home/folder2 -type f -mtime +5 -exec compress {} \;
month=$(cal | head -1 | grep -oP "[A-Za-z]+")
match=$(echo "${line}" | egrep -o 'run-parts (-{1,2}\S+ )*\S+')
groups
groups user
find . -name '*.js' -exec grep -i 'string to search for' {} \; -print
grep --help |grep recursive
grep --include=\*.{c,h} -rnw '/path/to/somewhere/' -e "pattern"
grep --exclude-dir={dir1,dir2,*.dst} -rnw '/path/to/somewhere/' -e "pattern"
grep -rnw '/path/' -e 'pattern'
grep --exclude=*.o -rnw '/path/to/somewhere/' -e "pattern"
find . -type 'd' | grep -v "NameToExclude" | xargs rmdir
find -type d -name a -exec rmdir {} \;
find . -name "a" -type d | xargs rmdir
grep -Ril "text-to-find-here" /
shopt -p | grep checkjobs
shopt | grep login
set | grep ^IFS=
find ./ -type f -iname "*.cs" -print0 | xargs -0 grep "content pattern"
find . | xargs grep "searched-string"
grep -R "texthere" *
set | grep ^fields=\\\|^var=
grep -e TEXT *.log | cut -d' ' --complement -s -f1
echo "$f" | grep -Eo '[0-9]+[.]+[0-9]+[.]?[0-9]?' | cut -d. -f1
echo "$f" | grep -Eo '[0-9]+[.]+[0-9]+[.]?[0-9]?' | cut -d. -f3
echo "$f" | grep -Eo '[0-9]+[.]+[0-9]+[.]?[0-9]?' | cut -d. -f2
grep -r -l "foo" .
grep -r "searched-string" .
find . -name "string to be searched" -exec grep "text" "{}" \;
TMPDIR=`dirname $(mktemp -u -t tmp.XXXXXXXXXX)`
dirname $(mktemp -u -t tmp.XXXXXXXXXX)
USERS=$(w | awk '/\/X/ {print $1}')
USERS=$(awk '/\/X/ {print $1}' <(w))
find -type f -exec grep -l "texthere" {} +
find . -type f -exec chmod 644 {} \;
find . -type d -exec chmod 755 {} \;
find /usr/local -name "*.html" -type f -exec chmod 644 {} \;
readlink -f $(which lshw)
$(dirname $(readlink -f $BASH_SOURCE))
find . -name '*.def' | sed 's/\(.*\)/\1.out/' | xargs touch
basename -a "${alpha[@]}"
awk '{print "result =",$0}' <(rev file)
rev file | awk '{print "result =",$0}'
join -11 -21 -o1.1,1.2,1.3,2.3 file1 file2
sed 's/$/ FAIL/' fail.txt | join -a 1 -e PASS -j 1 -o 1.1,2.2 list.txt -
join -j 1 -t : -o 2.1,2.2,2.4,1.3 <(sort empsal) <(sort empname)
join -v1 <(sort file1) <(sort file2)
join -o 1.1,1.2,1.3,1.4 -t, <(sort file1.csv) <(sort file2.txt)
join -o 1.1,1.2,1.3,1.4 -t, file1.csv file2.txt
join -j1 -o 2.1,2.2,1.2,1.3  <(sort test.1) <(sort test.2)
join -j1 file2 file1
crontab -l | sed -re '/# *change-enabled *$/s/^([^ ]+) [^ ]+/\1 7/' | crontab -
sudo ln -f "findpdftext" /usr/local/bin
ln -f secret_file.txt non_secret_file.txt
ln -f '/home/user/Musik/mix-2012-13/aesthesys~ I Am Free, That Is Why I'"'"'m Lost..mp3' '/home/user/Musik/youtube converted/aesthesys~ I Am Free, That Is Why I'"'"'m Lost..mp3'
sudo ln -s -f "/usr/local/bin/findpdftext" "/usr/local/bin/fpdf"
sudo ln -s -f "/usr/local/bin/findpdftext" "/usr/local/bin/fpt"
ln -sfn /other/dir new_dir
ln -sfn source_file_or_directory_name softlink_name
sudo ln -sTfv "$default_java_dir" "/usr/lib/jvm/default-java"
sudo ln -f -s $javaUsrLib/jdk1*/bin/* /usr/bin/
ln -sf "$(readlink -f "$1")" "$*"
ln -sfn "$c" "$lines"
sudo ln --symbolic --verbose --force "$pluginpath" "$pdir"
ln -sfn newDir currentDir
ln -sf $keyname     id_rsa
ln -f -s /apps/myapps/new/link/target mylink
ln -sf "$f" "~/my-existing-links/$(basename $f)"
ln -nsf $lastModified $SYMLINK_PATH
ln -nsf alpha_2 alpha
ln -nsf dir2 mylink
gzip -d --force * /etc
find . | xargs -i rm -f "{}"
find /tmp/* -atime +10 -exec rm -f {} \;
find . -maxdepth 1 -name "*.jpg" -size -50k -exec rm {} \;
find . -maxdepth 1 -name "*.jpg" -size -50k | xargs rm -f
find . -maxdepth 1 -type f -exec rm -f {} \;
find . -maxdepth 1 -type f -print0 | xargs rm -f
ssh -t somehost ~/bashplay/f
rm -rf "$(pwd -P)"/*
find /var/www -type d -print0 | xargs -0 chmod g+s
find . -name '*' | xargs rm
find . -name 'spam-*' | xargs rm
rm -rf *~important-file
ln --force --target-directory=~/staging ~/mirror/*
ln -sf '/cygdrive/c/Users/Mic/Desktop/PENDING - Pics/' '/cygdrive/c/Users/Mic/mypics'
rm --force "${temp}"
rm -f *.bak *~
rm -f /tmp/stored_exception /tmp/stored_exception_line /tmp/stored_exception_source
echo "$line" | column -t
cat file.txt | column -c 28 -s "\ "
date --date @120024000
echo " ${arr[@]/%/$'\n'}" | column
echo " ${arr[@]/%/$'\n'}" | sed 's/^ //' | column
awk '{for(i=3;i<=NF;i++){print $1,$2,$i}}' file | column -t
column -t -s $'\n' list-of-entries.txt
column -t -s '' list-of-entries.txt
mount | column -t
cat file | column -c 80
column -t -s' ' filename
column -t -s $'\t' list-of-entries.txt
column -t -s $'\t' FILE
cat file | column -t
column -t [file]
date -ud @1267619929
CDATE=$( date -d @"$timestamp" +"%Y-%m-%d %H:%M:%S" )
VARIABLENAME=$(date -d @133986838)
date -d @$TIMESTAMP
date -d @1267619929
ssh -f user@gateway -L 3307:1.2.3.4:3306 -N
ssh -f mysql_access_server -L 3309:sqlmaster.example.com:3306 -N
ssh -f -N -L localhost:12345:otherHost:12345   otherUser@otherHost
ssh -N -i <(echo "privatekeystuffdis88s8dsf8h8hsd8fh8d") -R 16186:localhost:8888 hello.com
ssh -f user@gateway -p 24222 -L 3307:1.2.3.4:3306 -N
ssh -L localhost:8000:clusternode:22 user@bridge
basename -- $0
basename $0
grep -Ff list1.txt list2.txt | sort | uniq -c | sort -n
pass=$(LC_CTYPE=C < /dev/urandom tr -cd [:graph:] | tr -d '\n' | fold -w 32 | head -n 1)
echo -e {{a..n},ñ,{o..z}}"\n" | nl
echo -e {{a..c},ch,{d..l},ll,{m,n},ñ,{o..z}}"\n" | nl
seq 1 10 | sort -R | tee /tmp/lst |cat <(cat /tmp/lst) <(echo '-------') \ <(tac)
MAPPER=$(mktemp -up /dev/mapper)
fn=$(mktemp -u -t 'XXXXXX')
tFile=$(mktemp --tmpdir=/dev/shm)
fifo_name=$(mktemp -u -t fifo.XXXXXX)
dig $domain
dig -t A $domain
who am i --ips|awk '{print $5}' #ubuntu 14
find / -size +10M -printf “%12s %t %h/%fn”
find /usr/bin  -type l  -name "z*" -exec ls  -l {} \;
find /home -type f -printf "%i@%p\n"
find . -type f -name '.*'
find / -type d -gid  100
find . -print0 | xargs -0 echo
find "$dir" -type f
find $dir -type f
find . -iname '*.page' -exec awk '{if(length($0) > L) { LINE=NR;L = length($0)}} END {print L"|"FILENAME":"LINE}' {} \; | sort
find . -name "*.rb" -type f -print0 | xargs -0 -n 2 echo
basename "$(pwd)"
pwd | awk -F / '{print $NF}'
header="$(curl -sI "$1" | tr -d '\r')"
dig +short "$domain"
$dig -x 8.8.8.8 | grep  PTR | grep -o google.*
reverse=$(dig -x $ip +short)
dig -x "$1" | grep PTR | cut -d ' ' -f 2 | grep google | cut -f 5
dig -x $IP | grep PTR | cut -d ' ' -f 2 | grep google | cut -f 5
dig -x 8.8.8.8| awk '/PTR[[:space:]]/ && /google/ {print $NF}'
cat 1.txt | xargs dig TXT
find / -ctime +3
grep ^Q File1.txt | cut -d= -f2- | sort | comm -23 - <(sort File2.txt)
find . -name 'filename' | xargs -r ls -tc | head -n1
echo "$data" | cut -f2 -d$'\n'
which find
echo "$path" | rev | cut -d"/" -f1 | rev
ps | grep `echo $$` | awk '{ print $4 }'
find . -type d -printf "%A@ %p\n" | sort -n | tail -n 1 | cut -d " " -f 2-
find /home/d -type f -name "*.txt" -printf "%s\n" | awk '{s+=$0}END{print "total: "s" bytes"}'
find . -iname "*.txt" -exec du -b {} + | awk '{total += $1} END {print total}'
find . -name "*.txt" -print0 |xargs -0 du -ch | tail -n1
find folder1 folder2 -iname '*.txt' -print0 | du --files0-from - -c -s | tail -1
find "$SEARCH_PATH" -name 'pattern' | rev | cut -d'/' -f3- | rev
find . -mindepth 1 -maxdepth 1 -type f -print0 | xargs -0 -I {} echo "{}"
find . -type f -name "*.php" -exec grep --with-filename -c "^use " {} \; | sort -t ":" -k 2 -n -r
find $1 -type f | wc -l
ps -ef | grep apache
find dir1 ! -type d |xargs wc -c
find dir2 ! -type d |xargs wc -c
find dir1 ! -type d -printf "%s\n" | awk '{sum += $1} END{print sum}'
find dir1 ! -type d -printf "%s\n" | awk '{sum += $1} END{printf "%f\n", sum}'
find dir1 ! -type d |xargs wc -c |tail -1
find . -type f -printf '%p %s\n'  | awk '{sum+=$NF}END{print sum}'
find . -type f -printf '%p %s\n' | awk '{ sum+=$2}; END { print sum}'
find path -type f -printf '%s\n' | awk '{sum += $1} END {print sum}'
NET_IP=`ifconfig ${NET_IF} | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1'`
ifconfig en0 | awk '$1 == "inet" {print $2}'
ifconfig en0 | awk '/inet addr/{print substr($2,6)}'
ifconfig en0 | awk '/inet addr/ {gsub("addr:", "", $2); print $2}'
ifconfig en1 | awk '{ print $2}' | grep -E -o "([0-9]{1,3}[\.]){3}[0-9]{1,3}"
ifconfig en1 | sed -n '/inet addr/s/.*addr.\([^ ]*\) .*/\1/p'
my_ip=$(ifconfig en1 | grep 'inet addr' | awk '{print $2}' | cut -d: -f 2)
ifconfig eth0 | awk '/inet /{sub(/[^0-9]*/,""); print $1}'
ifconfig eth0 | awk '/inet addr/{sub("addr:",""); print $2}'
ifconfig eth0 | awk '/inet addr/{sub(/[^0-9]*/,""); print $1}'
ifconfig eth0 | grep 'inet addr:' | cut -d: -f2 | awk '{print $1}'
ifconfig eth0 | grep -oP '(?<= inet addr:)[^ ]+'
ifconfig eth0 | grep inet | cut -d: -f2 | cut -d' ' -f1
ifconfig eth0 | grep addr: | awk '{ print $2 }' | cut -d: -f2
ifconfig  | grep 'inet addr:' | grep -v '127.0.0.1' | awk -F: '{print $2}' | awk '{print $1}' | head -1
ifconfig | grep 'inet addr:' | grep -v 127.0.0.1 | head -n1 | cut -f2 -d: | cut -f1 -d ' '
ifconfig | grep -E "([0-9]{1,3}\.){3}[0-9]{1,3}" | grep -v 127.0.0.1 | awk '{ print $2 }' | cut -f2 -d:
ifconfig | grep -A2 "venet0:0\|eth0" | grep 'inet addr:' | sed -r 's/.*inet addr:([^ ]+).*/\1/' | head -1
ip=$(ifconfig | grep -oP "(?<=inet addr:).*?(?=Bcast)")
ifconfig | awk -F':' '/inet addr/&&!/127.0.0.1/{split($2,_," ");print _[1]}'
ifconfig | grep 'inet' | grep -v '127.0.0.1' | awk '{print $2}' | sed 's/addr://'
ifconfig  | grep 'inet addr:'| grep -v '127.0.0.1' | cut -d: -f2 | awk '{ print $1}'
ifconfig | grep -oP "(?<=inet addr:).*?(?=  Bcast)"
ifconfig | grep -oP "(?<=inet addr:).*?(?=Bcast)"
ifconfig | grep -E "([0-9]{1,3}\.){3}[0-9]{1,3}" | grep -v 127.0.0.1 | awk '{ print $2 }' | cut -f2 -d:
ifconfig | grep ad.*Bc | cut -d: -f2 | awk '{ print $1}'
ifconfig eth0 | grep -Eo ..\(\:..\){5}
ifconfig eth0 | grep -o -E '([[:xdigit:]]{1,2}:){5}[[:xdigit:]]{1,2}'
ifconfig eth0 | head -n1 | tr -s ' ' | cut -d' ' -f5
ifconfig en0 | grep -Eo ..\(\:..\){5}
ifconfig en0 | grep -o -E '([[:xdigit:]]{1,2}:){5}[[:xdigit:]]{1,2}'
ifconfig eth0 | awk '/HWaddr/ {print $5}'
ifconfig eth0 | grep -Eoi [:0-9A-F:]{2}\(\:[:0-9A-F:]{2}\){5}
ifconfig eth0 | grep HWaddr | cut -d ' ' -f 9
ifconfig | grep -i hwaddr | cut -d ' ' -f9
ifconfig p2p0 | grep -o -E '([[:xdigit:]]{1,2}:){5}[[:xdigit:]]{1,2}'
ifconfig -a | awk '/^[a-z]/ { iface=$1; mac=$NF; next } /inet addr:/ { print mac }' | grep -o -E '([[:xdigit:]]{1,2}:){5}[[:xdigit:]]{1,2}'
ifconfig | awk '$0 ~ /HWaddr/ { print $5 }'
fg
fg 1
ifconfig | grep "inet addr:" | grep -v "127.0.0.1" | grep -Eo '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'  | head -1
fg 2
go=$(dig -x 8.8.8.8| awk '/PTR[[:space:]]/{print $NF}')
$dig -x 8.8.8.8 | grep  PTR | grep -o google.*
dig -x 8.8.8.8 | awk '/PTR[[:space:]]/{print $NF}'
dig -x 8.8.8.8 | grep PTR | cut -d ' ' -f 2 | grep google | cut -f 5
ifconfig | awk -F"[ :]+" '/inet addr/ && !/127.0/ {print $4}'
ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1'
ifconfig | sed -En 's/127.0.0.1//;s/.*inet (addr:)?(([0-9]*\.){3}[0-9]*).*/\2/p'
rest_cmd=$(shopt -p dotglob)
ifconfig en0 | grep inet | grep -v inet6
ifconfig eth0 | grep HWaddr
ifconfig eth0 | grep HWaddr
groups a b c d
find  / -name "apt" -ls
find . -name "*.pl" -exec ls -ld {} \;
find /path/to/base/dir -type d -exec chmod 755 {} +
find /path/to/base/dir -type d -print0 | xargs -0 chmod 755
find /path/to/base/dir -type f -exec chmod 644 {} +
find /path/to/base/dir -type f -print0 | xargs -0 chmod 644
find . -type d -name files -exec chmod ug=rwx,o= '{}' \;
find /home -xdev -samefile file1 | xargs ls -l
find . -name '*.php' | xargs wc -l | sort -nr | egrep -v "libs|tmp|tests|vendor" | less
groups $USERNAME | cut -d\  -f 1
cd -
cd $(ls -d */ | grep 1670)
cd `find . -maxdepth 1 -type d | grep 1670`
cd `ls -d */ | grep 1670`
cd /the/project/root//data
cd /tmp
cd /cygdrive/c/Program\ Files\ \(x86\)
cd "/cygdrive/c/Program Files (x86)"
cd '/cygdrive/c/Program Files (x86)/$dollarsign'
cd /some/where/long
cd "~"
cd `echo -n "~"`
cd "${dirs[-1]}"
cd $(echo $somedir | sed "s#^~#$HOME#")
cd $(dirname $(which ls))
cd $(which oracle | xargs dirname)
cd "$(ls -rd [0-9]*/ | tail --lines 1)"
cd -
source <(grep = file.ini | sed 's/ *= */=/g')
curl -s 'http://archive.ubuntu.com/ubuntu/pool/universe/s/splint/splint_3.1.2.dfsg1-2.diff.gz' | gunzip -dc | less
echo $(basename /foo/bar/stuff)
echo `basename "$filename"`
VAR=`dig axfr @dc1.localdomain.com localdomain.com | grep -i Lawler | awk '{ getline ; $1=substr($1,1,length($1)-1); print $1 ; exit }'`
dig $domain | grep $domain | grep -v ';' | awk '{ print $5 }'
yes n | gunzip file*.gz
shopt "$NGV" nullglob
ssh-keygen -Hf ~/.ssh/known_hosts
find . -type f -print0| xargs -0 grep -c banana| grep -v ":0$"
find /home/*/public_html/ -type f -iwholename "*/modules/system/system.info" -exec grep -H "version = \"" {} \;
find /var/www/vhosts/*/httpdocs/ -type f -iwholename "*/modules/system/system.info" -exec grep -H "version = \"" {} \;
find /home/*/public_html/ -type f -wholename *includes/constants.php -exec grep -H "PHPBB_VERSION" {} \;
find /var/www/vhosts/*/httpdocs/ -type f -wholename *includes/constants.php -exec grep -H "PHPBB_VERSION" {} \;
find /home/*/public_html/ -type f -iwholename "*/wp-includes/version.php" -exec grep -H "\$wp_version =" {} \;
find /var/www/vhosts/*/httpdocs/ -type f -iwholename "*/wp-includes/version.php" -exec grep -H "\$wp_version =" {} \;
find / -nouser -o  -nogroup
echo $c | crontab
set -e
ps -ef | grep myProcessName | grep -v grep | awk '{print $2}' | xargs kill -9
source "$DIR/incl.sh"
cat file-of-ips | xargs -n 1 -I ^ -P 50 ping ^
yes 0 | sed '1~2s/0/1/'
yes no
yes 1 | nl | tee /tmp/to
seq 1 10 | sed $': loop; n; n; a insert\nn; b loop'
fold -w30 longline | tr '\n' '|' | sed 's/|$/\n/'
sed -i "15i `hostname`" test.html
tac file | awk '/ScriptAlias/ && ! seen {print "new line"; seen=1} {print}' | tac
kill -9 \`pgrep myprocess\`
ln --symbolic --interactive $SCRIPT_DIR/$FILE
grep -R 'word' *.properties | more
source <(grep -v "mesg" /etc/bashrc)
grep "=" myfile | source /dev/stdin
FOO_NO_WHITESPACE="$(echo -e "${FOO}" | tr -d '[[:space:]]')"
source <( grep "marker" config.sh )
ssh -v -Y phil@192.168.0.14 -p 222
ssh -Y $ssh_user@$ssh_server
cut -d, -f1 file | uniq | xargs -I{} grep -m 1 "{}" file
join -t':' <(sort LN.txt) <(sort PH.txt) | join -t':'  - <(sort AD.txt)
awk 'NR==FNR{m[$1]=$2" "$3; next} {print $0, m[$1]}' file2 file1 | column -t
join file1 file2 | column -t
join -t, -a1 -a2 <(sort file1) <(sort file2)
join -t, <(sort test.1) <(sort test.2) | join -t, - <(sort test.3) | join -t, - <(sort test.4)
join -t, test.1 test.2 | join -t, - test.3 | join -t, - test.4
join -1 2 -2 1 <(sort +1 -2 file1) <(sort +0 -1 file2)
join -1 2 -2 1 -a1 <(cat -n file1.txt | sort -k2,2) <(sort file2.txt) | sort -k2 | cut --complement -d" " -f2
join <(sort -n A) <(sort -n B)
join <(sort aa) <(sort bb)
join <(sort aa) <(sort bb) | sort -k1,1n
paste <(uniq -f3 file | cut -f1,2) <(tac file | uniq -f3 | tac | cut -f3-)
join -o 1.2,1.3,2.4,2.5,1.4 <(cat -n file1) <(cat -n file2)
find  / -type d -name "apt" -ls
echo "abc-def-ghi-jkl" | rev | cut -d- -f-2 | rev
echo $path | rev | cut -d'/' -f-3 | rev
echo "0a.00.1 usb controller some text device 4dc9" | rev | cut -b1-4 | rev
ps -ef | grep myProcessName | grep -v grep | awk '{print $2}' | xargs kill -9
jobs -p | xargs kill -9
find /proc -user myuser -maxdepth 1 -type d -mtime +7 -exec basename {} \; | xargs kill -9
kill `pstree -p 24901 | sed 's/(/\n(/g' | grep '(' | sed 's/(\(.*\)).*/\1/' | tr "\n" " "`
nl -n ln log.txt | sed ...
fold -80 your_file | more
grep -o '1.' yourfile | head -n2
md5sum *.java | sort | uniq -d -w32
md5sum *.java | sort | uniq -d
find . -type f -ls | sort +7 | head -1
find . -name "*.pl" -exec ls -ld {} \;
find . -name *.txt -exec ls {} ;\
find . -name *.txt | egrep mystring
find . \( ! -name . -prune \) -name "*.c" -print
find /etc/nginx -name '*.conf' -exec echo {} ;
find . -name "*.html" -exec grep -lR 'base\-maps' {} \; | xargs grep -L 'base\-maps\-bot'
find . -name "*.log" -exec echo {} \;
ls -1 | xargs readlink -f
find  . -name '*.bak' -ls
find / \! -name "*.c" -print
find . -type f \( -name '*.c' -or -name '*.h' -or -name '*.cpp' \) -exec ls {} \;
find . -name *.gif -exec ls {} \;
find /usr /home  /tmp -name "*.jar"
find /home/bluher -name \*.java
find / -name "*.jpg" -print
find $HOME -name '*.ogg' -type f -exec du -h '{}' \;
find /home/kibab -name '*.png' -exec echo '{}' ';'
find . -type f -name '*.txt' -exec egrep -l pattern {} \;
find . -name "*.txt" -type f -print | xargs file | grep "foo=" | cut -d: -f1
find /etc -name "*.txt" -exec ls -l {} \;
find /etc -name "*.txt" -ls
find /etc -name "*.txt" | xargs -I {} ls -l {}
find /etc -name "*.txt" | xargs ls -l
ls -l $(find /etc -name "*.txt" )
find . -name "*.txt" -exec $SHELL -c 'echo "$0"' {} \;
find . -name "*.txt" -print
find . -name '*.txt' -exec echo "{}" \;
find . -name *.txt -exec ls {} \;
find . -name '*.txt' -print0|xargs -0 -n 1 echo
find / \( -type f -or -type d \) -name \*fink\* -ls
find . -name "*fink*" |xargs ls -l
find . \( -name '*jsp' -o -name '*java' \) -type f -ls
find . -name '*.[ch]' -print0 | xargs -r -0 grep -l thing
find . -name '*.[ch]' | xargs grep -l thing
find . -name *.gif -exec ls {} \;
find . -name "*.jpg" -exec ls {} \;
find . -name "*.jpg" -print0 | xargs -0 ls
find . -name "*.jpg" -exec ls {} +
find . -name .svn -exec echo {} \;
find . -name .svn -exec ls {} \;
more /etc/hosts | grep '[[:space:]]*'`hostname`'[[:space:]]*' | awk '{print $1}'
more /etc/hosts | grep `hostname` | awk '{print $1}'
find /etc -type f -exec cat '{}' \; | tr -c '.[:digit:]' '\n'  | grep '^[^.][^.]*\.[^.][^.]*\.[^.][^.]*\.[^.][^.]*$'
find / -type f \( -perm -4000 -o -perm -2000 \) -ls
crontab -l | grep -v "^#" | awk '{print $6}'
find /home/jassi/ -name "aliencoders.[0-9]+" -exec ls -lrt {} + | awk '{print $9}'
find /home/jassi/ -name "aliencoders.[0-9]+" | xargs -r ls -lrt | awk '{print $9}'
find /home/jassi/ -name "aliencoders.[0-9]+" | xargs ls -lrt | awk print '$9'
find /home/jassi/ -name "aliencoders.[0-9]+" |& xargs ls -lrt | awk '{print $9}'
echo $PATH | tr ':' '\n' | xargs -I {} find {} -maxdepth 1 -type f -perm '++x'
find . -type l -printf "%Y %p\n" | grep -w '^N'
crontab -l
crontab -l | awk '$1 == "*" || $1 ~ /^[0-9]+$/ && $1 < 15 {print}'
crontab -l | egrep "word"
crontab -l | grep 'word'
cat /etc/passwd | sed 's/^\([^:]*\):.*$/crontab -u \1 -l 2>\&1/' | grep -v "no crontab for" | sh
set | egrep '^[^[:space:]]+ [(][)][[:space:]]*$' | sed -r -e 's/ [(][)][[:space:]]*$//'
find . -type d
find . -type d
find -maxdepth 1 -type d | awk -F"./" '{print $2}'
find . -maxdepth 1 -type d -exec ls -dlrt {} \;
find . -type d -maxdepth 1 -exec ls -dlrt {} \;
find "$topdir" -name '*.py' -printf '%h\0' | xargs -0 -I {} find {} -mindepth 1 -maxdepth 1 -name Makefile -printf '%h\n' | sort -u
find . -type d -exec ls -dlrt {} \;
find . -empty -exec ls {} \;
find . -type f -empty
find . -maxdepth 1 -empty
find ./in_save/ -type f -maxdepth 1| more
set
env | grep '^\(GO\|HOME=\|PATH=\)'
env | grep '^\(GOBIN\|PATH=\)'
env | grep '^GOROOT'
sudo env |grep USER
env | grep ipo | awk 'BEGIN {FS="="} ; { print $1 } '
env | sed -n /"$USERNAME"/p
env | grep ^PATH
env | awk -F= '/[a-zA-Z_][a-zA-Z_0-9]*=/ { if (!system("[ -n \"${" $1 "+y}\" ]")) print $1 }' | sort | uniq
find . -iname '.note' | sort -r
tree -L 2 -fi
ls -alrt `pwd`/*
find . -print | xargs ls
ls -l --time-style=long-iso | sort -k6
find /home -name Trash -exec ls -al {} \;
find -print0 | xargs -0 ls
find | xargs ls
ls `pwd`/*
tree -af
find . -type f -size +10000 -exec ls -al {} \;
find . -print -o -name SCCS -prune
find . -mmin -60 -ls
find . -mmin -60 | xargs -r ls -l
find . -mmin -60 | xargs -r ls -ld
find . -mindepth 1 -mmin -60 | xargs -r ls -ld
find /home/bozo/projects -mtime 1
find /home/bozo/projects -mtime 1
find /home/bozo/projects -mtime -1
find /home/bozo/projects -mtime -1
find /var/www -cmin -10 -printf "%c %pn"
ls -1 | tr '\n' ','
ls -1b | tr '\n' ';'
ls -m | tr -d ' ' | tr ',' ';'
file * | grep ASCII
find / -newer ttt -user wnj -print
find / -type f -name "*" -newermt "$newerthan" ! -newermt "$olderthan"  -printf '%T@ %p\n' | sort -k 1 -n | sed 's/^[^ ]* //'
find / -type f -name "*" -newermt "$newerthan" ! -newermt "$olderthan" -printf "%T+\t%p\n" | sort | awk '{print $2}'
find / -type f -name "*" -newermt "$newerthan" ! -newermt "$olderthan" -ls
find / \! \( -newer ttt -user wnj \) -print
find / \( -newer ttt -or -user wnj \) -print
find . -maxdepth 2  -type f -exec ls -l {} \;
find . -maxdepth 2  -type f -print0 | xargs -0 -n1 ls -l
find test -print | grep -v '/invalid_dir/'
find /hometest -name Trash -exec ls -s {} \;
find /myfiles -exec ls -l {} ;
find /var/ -size +10M -exec ls -lh {} \;
find /var/ -size +10M -ls
find . -path './src/emacs' -prune -o -print
find . -print0 | xargs -0 -l -i echo "{}";
find . -type f -print | xargs ls -l
find . -size +1000k
find -newermt "mar 03, 2010 09:00" -not -newermt "mar 11, 2010" -ls
find ! -newermt "apr 01 2007" -newermt "mar 01 2007" -ls
find -mmin +60
find -newermt "mar 03, 2010" -ls
find -newermt yesterday -ls
find -mmin 60
find . -mmin 60 -print0 | xargs -0r ls -l
find . -mmin 60 | xargs '-rd\n' ls -l
find . -mmin -60 |xargs ls -l
find . -name '*foo*' -exec ls -lah {} \;
find . -name FOLDER1 -prune -o -name filename -print
find . -size +10000c -size -32000c -print
find . -type f -atime +30 -print
find /home/musicuser/Music/ -type f  -iname "*$1*" -iname "*$2*" -exec echo {} \;
find $ARCH1 -ls
find $FULFILLMENT -ls
find . -type f | xargs ls
find -E . -type f -regex '.*(c|h|cpp)$' -exec ls {} \;
find . -type f -regex '.*\(c\|h\|cpp\)' -exec ls {} \;
find . -type f -regex '.*\.\(c\|h\|cpp\)' -exec ls {} \;
find . -type f -size +10000000 -print|xargs ls -ld|more
find . -size +10M -exec ls -ld {} \;
find . -type f |xargs ls -lS |head -20 | awk '{print $9, $5}'
find . -type f -printf '%s %p\n'
find . -type f -print0 | xargs -0 ls
find `pwd` -mtime -1 -type f -print
find $(pwd)/ -type f
find `pwd` -name .htaccess
find . -name "someFile" -printf "%p:%T@\n" | sort -t : -k2
find / -type f -name "*" -newermt "$newerthan" ! -newermt "$olderthan" -printf "%T+\t%p\n" | sort
find * -type f | xargs md5sum | sort | uniq -Dw32
find * -type f -print -o -type d -prune
find / -print
find $dir_name -size $sizeFile -printf '%M %n %u %g %s %Tb %Td %Tk:%TM %p\n'
find /data1/Marcel -size +1024 \( -mtime +365 -o -atime +365 \) -ls
find /data1/Marcel -size +1024  \( -mtime +365 -o -atime +365 \) -ls -exec file {} \;
find /myfiles -exec ls -l {} ;
find -ls
find .
find . -ls
find . -print
find . -ls | tr -s ' ' ,
find -print0 | xargs --null
find . -regextype posix-egrep -regex ".+\.(c|cpp|h)$" -print0 | xargs -0 -n 1 ls
find . -regextype posix-egrep -regex ".+\.(c|cpp|h)$" | xargs -n 1 ls
find . -regextype posix-egrep -regex '.+\.(c|cpp|h)$' -print0 | grep -vzZ generated | grep -vzZ deploy | xargs -0 ls -1Ld
find . -regextype posix-egrep -regex '.+\.(c|cpp|h)$' -not -path '*/generated/*' -not -path '*/deploy/*' -print0 | xargs -0 ls -L1d
find . -ls | awk '{printf( "%s,%s,%s,%s,%s,%s,%s,%s %s %s,%s\n", $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11 )}'
find . -ls|awk 'BEGIN{OFS=","}$1=$1'
find . -name 'FooBar' -print0 | xargs -0
find . -print0 | grep --null 'FooBar' | xargs -0
find . -type f -fprintf /tmp/files.txt "%i,%b,%M,%n,%u,%g,%s,%CY-%Cm-%Cd %CT,%p\n"
find ~/Library -name '* *' -exec ls {} \;
ls | tr  "\n" " "
set | grep " () $" | cut -d' ' -f1
set | grep " ()"
find . -maxdepth 1 -type f -name '.*' -printf '%f\0'
find -depth -type d |sed 'h; :b; $b; N; /^\(.*\)\/.*\n\1$/ { g; bb }; $ {x; b}; P; D'
find . -type d -links 2
find -depth -type d |sed 'h; :b; $b; N; /^\(.*\)\/.*\n\1$/ { g; bb }; $ {x; b}; P; D'
find . -type d -links 2
find . -type d | sort | awk '$0 !~ last "/" {print last} {last=$0} END {print last}'
mount
mount -l | grep 'type nfs' | sed 's/.* on \([^ ]*\) .*/\1/'
mount -l -t nfs4
find . -type f ! -size 0
find ~/junk   -name "*" -exec ls -l {} \;
ls -d */ | cut -f1 -d'/'
find /data/ -name 'command-*-setup' | sort -t - -V -k 2,2
find . -name '*.php' -print0 | xargs -0 wc -l | sort -nr | egrep -v "libs|tmp|tests|vendor" | less
ps -ef
find / -type f -user root -perm -4000 -exec ls -l {} \;
find -type f -mtime -60
find . -mmin -60 -type f -exec ls -l {} \;
find . -mmin -60 -type f -ls
find . -mmin -60 -type f | xargs ls -l
find . -type f -mmin -60 -print0 | xargs -r0 ls -l
find . -type f -mmin -60 | xargs -r ls -l
find /var/www -cmin -10 -type f -printf "%c %pn"
find ~ -type f -mmin -90 | xargs ls -l
find ~ -type f -mtime +1825 |xargs -r ls -l
find ~ -type f -mtime +1825 |xargs ls -l
find / -type f -exec echo {} \;
find . -type f -exec ls -l '{}' \;
find . -type f -print0 | xargs -0 ls -l
find . -type f | xargs ls -l
find . -mtime 0 -type f -ls
find . -mmin -60 -type f -exec ls -l {} +
find /usr /bin /sbin /opt -name "$1*" -type f -ls
find -type f -mtime 61 -exec ls -ltr {} \;
find . -type f -exec grep -li '/bin/ksh' {} \;
find . -type f -print | xargs grep -li 'bin/ksh'
find . -type f -print | xargs -n 1
find . -type f -print0 | xargs -0 -n 1
find . -type f -print0 | xargs -0 ls -l
find . -type f | xargs ls -l
find . -type l | xargs -I % ls -l % | grep targetfile.txt
find . -name "*.c" | xargs -n3
find . -name "*.c" | xargs -n3 echo
find . -maxdepth 1 -empty
find . -type l
find -name '*.class' -printf '%h\n' | sort -u
env | grep ".*X.*"
env | awk -F "=" '{print $1}' | grep ".*X.*"
env | awk -F= '{if($1 ~ /X/) print $1}'
find ~
find . -empty -exec ls {} \;
comm -12 <(find ~/bin/FilesDvorak/.* -maxdepth 0) <(find ~/.PAST_RC_files/.*)
env -i
find /prog -type f -size +1000 -print -name core -exec rm {} \;
find /prog -type f -size +1000 -print -name core -exec rm {} \;
find . -type d | sort | awk '$0 !~ last "/" {print last} {last=$0} END {print last}'
comm -3 <(sort -un f1) <(sort -un f2) | tr -d '\t'
od -cvAnone -w1
cat <(ls 1 | sort -u) <(ls 2 | sort -u) | uniq -d
$(cat myfile)
ls | more
crontab -l
ls -1p | grep '/$' | sed 's/\/$//'
ls -d */|sed 's|[/]||g'
find . -type d
du -sh */ | sort -n
du -a --max-depth=1 | sort -n
du -h --max-depth=0 * | sort -hr
du -smc * | sort -n
du -s * | sort -n
pwd | cut -b2- | tr '/' '\n'
pwd | cut -f 1- -d\/ --output-delimiter=$'\n'
grep -o . file | sort -f | uniq -ic
grep -o . filename | tr '[:upper:]' '[:lower:]' | sort | uniq -c | sort -nr
grep -o . file | sort | uniq -c
grep -o . filename | sort | uniq -c | sort -nr
find . -depth -empty -type d
set | grep -P '^\w*X\w*(?==)' | grep -oP '(?<==).*'
set | grep -P '^\w*X\w*(?==)'
env | sed 's/;/\\;/g'
set | cut -d= -f1 | grep X
set | grep -oP '^\w*(?==)' | grep X
set | awk -F "=" '{print $1}' | grep ".*X.*"
set | grep -oP '^\w*X\w*(?==)'
find `pwd` -group staff -exec find {} -type l -print ;
find . -perm /111 -type f | sed 's#^./##' | sort | diff -u .gitignore -
gzip -l $i
gzip -l compressed.tar.gz
ls -ald `which c++`
tree -L 2
find . -print | xargs ls -gilds
find . -name "*.html"
find /path/to/directory -type f -size +1024k -exec ls -lh {} \; | awk '{ print $8 ": " $5 }'
comm -23 <(ls dir1 |sort) <(ls dir2|sort)
sort <(ls one) <(ls two) | uniq -u
sort <(ls one) <(ls two) | uniq -d
find . \( ! -name . -prune \)
find . \( -path './*' -prune \)
find -ls
find . -not -iwholename '*/.git/*'
find . -size 1234c
find . -exec echo {} ;
find . -perm 766 -exec ls -l {} \;
find /var/ -size +10M -exec ls -lh {} \;
find /var/ -size +10M -ls
find /var/log -size +10M -ls
find /tmp /var/tmp -size +30M -mtime 31 -ls
find /tmp /var/tmp -size +30M -mtime 31 -print0 | xargs -0 ls -l
find . -size +9M
find `pwd` -name "accepted_hits.bam" | xargs -i echo somecommand {}
find . -size -9k
find ${CURR_DIR} -type f \( -ctime ${FTIME} -o -atime ${FTIME} -o -mtime ${FTIME} \) -printf "./%P\n"
find . -type f -exec ls -s {} \; | sort -n -r
ls -b
find / -mindepth 3  -name "*log"
find . -type f |xargs ls -lS |head -20
find . -name something -print | head -n 5
find . -type f -name "*.txt" ! -path "./Movies/*" ! -path "./Downloads/*" ! -path "./Music/*" -ls
find -type f |  grep -P '\w+-\d+x\d+\.\w+$' | sed -re 's/(\s)/\\\1/g' | xargs ls -l
find /somelocation/log_output -type f -ctime +40 -exec ls -l {} \;
joblist=$(jobs -l | tr "\n" "^")
find . -mindepth 2 -maxdepth 2 -type d -ls
find . -mindepth 2 -maxdepth 2 -type d -printf '%M %u %g %p\n'
find . -type f -not -name '.*' -mtime +500 -exec ls {} \;
find . -maxdepth 1 -empty -not -name ".*"
find . -type f -iname "*.php"  -exec file "{}" + | grep CRLF
find /var/www
find /var/www | more
find . -type f  -perm 777 -exec ls -l {} \;
find . -type f  -perm a=rwx -exec ls -l {} \;
find / -type f -user root -perm -4000 -exec ls -l {} \;
find . -maxdepth 1 -type d -exec ls -ld "{}" \;
find . -maxdepth 1 -type d -print0 | xargs -0 ls -d
du -a /var | sort -n -r | head -n 10
find teste1 teste2 -type f -exec md5 -r {} \; | sort
ls "`pwd`/file.txt"
ls /usr/bin | more
more <( ls /usr/bin )
find . -path ./src/emacs -prune -o -print
find . -path ./dir1  -prune -o -print
find . -path ./dir1\*  -o -print
find . -path ./dir1\*  -prune -o -print
find . -print -name dir -prune
echo $(ls $(pwd))
find /path -type f -iname "*.ext" -printf "%h\n"
find . -name "file.ext" -execdir pwd ';'
find `pwd` -name "file.ext" -exec dirname {} \;
find `pwd` -name file.ext |xargs -l1 dirname
mount | grep '^/dev/' | sed -E 's/([^ ]*) on ([^ ]*) .*/"\2" is located on "\1"/'
crontab -u apache -l
find . -type f -print | xargs grep -il '^Subject:.*unique subject'
find . -type f -print0 | xargs -0 grep -il '^Subject:.*unique subject'
find . -name bin -prune -o -name src -prune -o -type f -print | xargs egrep -il '^From:.*unique sender'
gzip -l archive.tar.gz
find /etc -type f -print | xargs grep -il old1\.old2\.co\.com
find /etc -type f -print | xargs grep -il '128\.200\.34\.'
find /PATH_to_SEARCH -ls | sort -n | awk '!seen[$1]++'
tree -dfi -L 1 "$(pwd)"
tree -dfi "$(pwd)"
find . -type f -ls | sort -nrk7 | head -1 #unformatted
find . -type f -name '*.gz' -printf '%s %p\n'|sort -nr|head -n 1
find /foldername | sort -n | tail -1
find $DIR -type f -printf "%T@ %p\n" | sort -n | cut -d' ' -f 2 | tail -n 1
find `pwd` -name "file.ext" -printf "%f\n"
find . -type f -exec basename {} \;
find . -maxdepth 1 -mindepth 1 -type d -printf '%f\n'
find . -mindepth 1 -maxdepth 1 -type d -printf "%P\n"
echo "The quick brown fox jumps over the lazy dog" | grep -o . | sort | uniq -c | sort -nr
find -type f -exec grep -l "texthere" {} +
find . -type f | grep -o -E '\.[^\.]+$' | sort -u
find / -name *.class -printf '%h\n' | sort --unique
find /root_path -type f -iname "*.class" -printf "%h\n" | sort -u
cut -d/ -f1-2 | cut -d/ -f2- | sort | uniq -c
cut -f $FIELD * | sort| uniq -c |sort -nr
find /usr/bin  -type l  -name "z*" -exec ls  -l {} \;
find /usr/bin -type  l  -name "z*" -ls
md5sum *.txt | cut -d ' ' -f 1 | sort -u
fold -w3 "$1" | sort | uniq -c | sort -k1,1nr -k2
ls -1 /tmp/hashmap.$1
ls -d /home/alice/Documents/*/
ls -d */
ls -d ./*/
ls /home/dreftymac/*
ls -1 | paste -sd "," -
ls -1 | tr '\n' ',' | sed 's/,$/\n/'
ls -m
ls | sed '$!s/$/,/' | tr -d '\n'
ls | xargs -I {} echo {}, | xargs echo
ls -1 | awk 'ORS=" "'
ls -l /lib*/ld-linux*.so.2
apropos -r '.*'
ls -mR * | sed -n 's/://p'
ls -d */ | cut -f1 -d'/'
ls -d */|sed 's|[/]||g'
ls -d ./*/                     ### more reliable BSD ls
ls -d -- */                    ### more reliable GNU ls
ls -d1 */ | tr -d "/"
ls /home/dreftymac/
ls -d -1 $PWD/**/*
zless MyFile
$ ls
ls -l /proc/self/fd/
ls -l /bin/echo
pstree -p 1782 | sed 's/-/\n/g' | sed -n -e 's/.*(\([0-9]\+\)).*/\1/p'
bind -f ~/.inputrc
find $HOME -iname '*.ogg' -size +100M
find $HOME -iname '*.ogg' -type f -size -100M
find /usr/share/doc -name "copyright"
find . -name "*.csv"
find . -name "*.csv" -print
find . -name "*.csv" -print0
find . -type f -name "*.csv"
find . -name '*.mov'
find . -name "*.txt"
find . -name "*.txt"
find $HOME -iname '*.ogg' -atime +30
find ~ -name readme.txt
find . -name "needle.txt"
find -not -name "*testfileasdf*"
find -name "*testfile*"
find -iname "*TESTFILE*"
find . -maxdepth 4 -name 'restore.php'
find / -name passwd
find / -samefile passwd
find . -inum 211028 -exec mv {} newname.dir \;
find -name file1
find / -path /proc -prune -o -nouser -o -nogroup
find /u/bill -amin +2 -amin -6
find $HOME -atime +30
find . -ctime -1 -print
find . -user my_user -perm -u+rwx
find /var/www -name logo.gif
find /usr -type l
find lpi104-6 research/lpi104-6 -type l
find / -name httpd.conf
find / -path /proc -prune -o -perm -2 ! -type l -ls
libdir=$(dirname $(dirname $(which gcc)))/lib
awk -F, 'NR==1 {gsub(/"/,"",$3);print $3}' "$(dirname $(readlink -f $(which erl)))/../releases/RELEASES"
which bzip2
find /home -type f -size +100M -delete
ssh -i ./middle_id.pem -R 22:localhost:2222 middleuser@middle.example.org
ssh -i ./device_id.pem -p 2222 deviceuser@middle.example.org
ssh -i ~/path/mykeypair.pem ubuntu@ec2-XX-XXX-XXX-XXX.us-west-2.compute.amazonaws.com
ssh -X whoever@whatever.com
ssh -q $HOST "[[ ! -f $FILE_PATH ]] && touch $FILE_PATH"
ssh -i id_rsa host
find -name filename
find / -maxdepth 2 -name testfile.txt
find . -name “*.jpg”
find / -name “*.jpg”
find / -mindepth 3  -name "*log"
find / -maxdepth 3  -name "*log"
find / -perm /g=s
find / -perm +4000
find .  -type f -print|xargs file|grep -i text|cut -fl -d:    | xargs grep regexp
find / -name 'my*'
find -mtime +2
find -mtime +2 -mtime -5
find -name Cookbook -type d
find /usr /home -name Chapter1 -type f
find . -perm -070 -print
find / -perm +6000 -type f
find  / -name "[a-j]*" -print
find . -name "search"
find . -type f  -perm 777 -exec ls -l {} \;
find . -type f  -perm a=rwx -exec ls -l {} \;
find /var/spool/postfix/{deferred,active,maildrop,incoming}/ -type f
find /home/dm -name "*uniform*"
find . \( -name "my*" -o -name "qu*" \) -print
IP=$(dig +short myip.opendns.com @resolver1.opendns.com)
finger vivek
finger `whoami`
chmod +x bar
chmod +x file.sh
mkdir -p $(seq -f "weekly.%.0f" 0 $WEEKS_TO_SAVE)
mkdir $(seq -f "$HOME/Labs/lab4a/folder%03g" 3)
mkdir ~/Labs/lab4a/folder{1,2,3}
mkdir ~/Labs/lab4a/folder{1..3}
mkdir -p folder$( seq -s "/folder" 999 )1000
dig stackoverflow.com
dig -f /path/to/host-list.txt
wget --post-data="PiIP=$(hostname -I)" http://dweet.io/dweet/for/cycy42
ls -d */ | xargs -iA cp file.txt A
echo dir1 dir2 dir3 | xargs -n 1 cp file1
cp -R SRCFOLDER DESTFOLDER/
cp -r dir1/ ~/Pictures/
cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1 | xargs mkdir
mkdir $(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
find . | grep -E -v '\.png$|\.class$' | vim -
find . | grep -v "\.png$" | grep -v "\.class$" | vim -
find . -type f -noleaf -links +1 -printf "%n %i %f\t%h\n" | sort | less
find . -type d | xargs -I "{x}" mkdir "{x}"/new-dir
find . -maxdepth 1 -type d | xargs -I "{x}" mkdir "{x}"/new-dir
find . -type d -print0 | xargs -0 chmod go+rx
find . -type f -print0 | xargs -0 chmod go+r
find bugzilla -type d -exec chmod 775 {} \;
mkdir -p es/LC_MESSAGES
mkdir --parents ./es_MX.utf8/LC_MESSAGES
mkdir "$@" |sed -e"s/mkdir: created directory /$USER created folder /"
mkdir -v "$@" | sed 's/mkdir: created directory /jar-jar: yea, weesa gotta /'
mkdir --parents ./es_MX.utf8/LC_MESSAGES
mkdir /tmp/A /tmp/B /tmp/C "/tmp/ dir with spaces"
mkdir 3/foo 3/bar 3/baz
mkdir Labs/lab4a/{folder1,myfolder,foofolder}
mkdir a b c d e
mkdir -p a/b/c
mkdir bravo_dir alpha_dir
mkdir -p es/LC_MESSAGES
mkdir foo bar
mkdir -p foo/bar/baz
mkdir mnt point
mkdir -p project/{lib/ext,bin,src,doc/{html,info,pdf},demo/stat/a}
mkdir -p tmp/real_dir1 tmp/real_dir2
mkdir -p ~/foo/bar/baz ~/foo/bar/bif ~/foo/boo/bang
mkdir -p path2/{a..z}
mkdir -p $(echo -e $1 |sed $'s/\r//')
cat a.txt | grep / | sed 's|/[^/]*$||' | sort -u | xargs -d $'\n' mkdir -p
mkdir -p `cut -f2 -d"&" filetypes.txt | sed 's/[ ,:]//g' | sort -u`
mkdir -p ${raw_folder} ${split_folder}
find src/ -type d -exec mkdir -p dest/{} \;
cat folder_list.txt | xargs mkdir
cat file1 |xargs -I {} dirname "{}"| sort -u | xargs -I{} mkdir -p "{}"
find . -type f -iname \*.mov -printf '%h\n' | sort | uniq | xargs -n 1 -d '\n' -I '{}' mkdir -vp "/TARGET_FOLDER_ROOT/{}"
mkdir -p $2
mkdir -p "$TARGET_PATH"
mkdir -p /my/other/path/here
mkdir -p /my/other/path/here/
mkdir -p ./some/path
mkdir -pv /tmp/boostinst
mkdir -p /tmp/test/blah/oops/something
mkdir -p directory{1..3}/subdirectory{1..3}/subsubdirectory{1..2}
mkdir -p x/p/q
mkdir -p `dirname /full/path/to/file.txt`
mkdir /cpuset
sudo mkdir /data/db
mkdir /etc/cron.15sec
mkdir /etc/cron.5minute
mkdir /etc/cron.minute
mkdir /path/to/destination
mkdir /tmp/foo
mkdir /tmp/googleTestMock
rsync -aq --rsync-path='mkdir -p /tmp/imaginary/ && rsync' file user@remote:/tmp/imaginary/
mkdir /tmp/new
sudo mkdir /var/svn
mkdir TestProject
mkdir aaa
mkdir aaa/bbb
mkdir alpha_real
mkdir backup
mkdir certs/
mkdir destdir
mkdir -p dir
mkdir dir1
mkdir -m 777 dirname
mkdir foo
mkdir -p foo
mkdir mybuild
mkdir new_dir
mkdir saxon_docs
mkdir subdirectory
mkdir tata
mkdir temp
mkdir testExpress
mkdir ~/log
mkdir ~/practice
mkdir ~/public_html
mkdir ~/temp
mkdir -p `file -b --mime-type *|uniq`
mkdir -p -- "$(dirname -- "$f")"
mkdir -p "$FINALPATH"
mkdir -p ~/temp/bluecove/target/
mkdir $dir
mkdir ${archive%.tar*}
mkdir .hiddendir
find debian/fglrx-amdcccle/usr/lib/fglrx/bin/ -type f | xargs chmod a+x
touch .bash_profile
set -a
find /directory1/directory2/ -maxdepth 1 -type f | sort | tail -n 5 | xargs md5sum
find /home/d -type f -name "*.txt" -printf "%s\n" | awk '{s+=$0}END{print "total: "s" bytes"}'
find folder1 folder2 -iname '*.txt' -print0 | du --files0-from - -c -s | tail -1
find . -name "*.txt" -print0 |xargs -0 du -ch
find . -name "*.txt" -print0 |xargs -0 du -ch | tail -n1
find . -iname "*.txt" -exec du -b {} + | awk '{total += $1} END {print total}'
sort -m a b c d e f g h i j | more
sort -m *.txt | split -d -l 1000000 - output
sort --merge file*.txt | split -l 100000 - sorted_file
sort -m *.$suffix
sort -m _tmp* -o data.tsv.sorted
join -t: <(sort file1) <(sort file2)
join -o 1.1,1.2,1.3,2.1,2.2,2.3 -j2 <(sort -k2 file1) <(sort -k2 file2)
join -t, -o 1.2,2.2,2.3 -a 1 -a 2 -e 'no-match' - <(sort file1.txt)
join -t, -o 1.2,2.2,2.3 -a 1 -e 'no-match' - <(sort file1.txt)
sort -m <(zcat $part0 | sort) <(zcat $part1 | sort) ...
join -j2 <(sort -k2 file1) <(sort -k2 file2)
paste -d, -s file
paste -s -d","
grep -v '^$' | paste -s -d"," -
join -t, <(sort file1) <(sort file2)
join -1 2 -2 1 text.txt codes.txt
paste file_1 file_2 | column -s $'\t' -t
paste file{1,2,3,4} | sed -e 's/\t/ \t/g' | column -t -s$'\t'
join -t, -o 1.1,1.2,2.3 in1 in2
join -t, in1 in2
join -t, -o 1.1,1.2,2.3 -a1 in1 in2
paste <(head -"$lc" current.txt) <(tail -"$lc" current.txt) | column -t -o,
sed -i 's/foo/bar/g' file
sed -i 's/foo/bar/' file
watch -n 0.1
top -p 18884 -p 18892 -p 18919
mount /dev/shm
mount /path/to/device /path/to/mount/location -o loop
mount /path/to/device /path/to/mount/location -o loop -t vfat
mount /tmp/loop.img /mnt/image -o loop
mount /windows
mount -t cpuset none /cpuset/
sudo mount device_name mount_point
sudo mount /dev/xvdf /vol -t ext4
sudo mount /dev/xvdf1 /vol -t ext4
mount -t ntfs -o ro /dev/mapper/myldm /mnt
mount -t ntfs-3g /dev/mapper/myvolume /media/volume
mount -t proc none /var/snmp3/proc
mount none -t tmpfs /path/to/dir
sudo mount -t vfat -o rw,auto,user,fmask=0000,dmask=0000 /dev/sda7 /mnt/my_partition
mount -o ro,loop,offset=$OFFSET -t auto $IMAGE /media/$DEST
mount -t ntfs
sudo mount -a
sudo mount -o loop /dev/loop0 test
mount -L WHITE /mnt/WHITE
mount -L WHITE /mnt/WHITE -o rw
mount -L WHITE /mnt/WHITE -o rw,uid=test,gid=test
sudo mount -t cifs -o username=me,password=mine //192.168.0.111/serv_share /mnt/my_share
mount -t cifs //server/source/ /mnt/source-tmp -o username=Username,password=password
mount -t smbfs -o soft //username@server/share /users/username/smb/share
mount -t linprocfs none /proc
sudo mount -t vboxsf D:\share_folder_vm \share_folder
sudo mount -t vboxsf myFileName ~/destination
mount -obind /etc /tmp/sarnold/mount_point/
sudo mv $PHANTOM_JS /usr/local/share
mv /usr/bin/openssl /root/
mv -nv caniwrite /usr/local/bin
mv -f file.txt /var/lib/docker/devicemapper/mnt/$CONTAINER_ID/rootfs/root/file.txt
mv -f file.txt /var/lib/docker/aufs/mnt/$CONTAINER_ID/rootfs/root/file.txt
mv -t target file1 file2 ...
sudo mv phantomjs-1.8.1-linux-x86_64.tar.bz2 /usr/local/share/.
mv tobecopied/tobeexclude tobeexclude;
mv tobecopied/tobeexcluded .
find /var/www/ -path '*wp-admin/index.php' -exec mv {} $(dirname {})/index_disabled
find "${S}" -name '*.data' -exec mv '{}' "${S}/data/" \;
find /path/to/folders/ -name \*.emlx -print0 | xargs -0 -I {} mv {} ./Messages/
find /foo/bar -name '*.mp4' -exec mv -t /some/path {} +
find /foo/bar -name '*.mp4' -print0 | xargs -0 mv -t /some/path {}
find ${INPUT}/ -name "*.pdf" -exec mv '{}' '{}'.marker ${OUTPUT} \;
find . -iname "*.php~" -exec mv "{}" /mydir +;
find . -iname "*.php~" -exec mv {} /mydir \;
find /path -type f -name "10*jpg" | sed 's/.*/mv &/' | sed 's/mv \(.*\/\)\(.[^/]*\)/& \120\2/' | sh
find . -name '*~' -print 0 | xargs -0 -I % cp % ~/backups
find sourceDir -mindepth 1 -type d  -exec mv -t destDir "{}"  \+
find sourceDir -mindepth 1 -type d  -print0 | xargs -0 mv --target-directory=destDir
find ./ -maxdepth 1 -name "some-dir" -type d -print0 | xargs -0r mv -t x/
find . -depth -type d -mtime 0 -exec mv -t /path/to/target-dir {} +
find . -type d -mtime -0 -exec mv -t /path/to/target-dir {} +
find . -type d -mtime -0 -print0 | xargs -0 mv -t /path/to/target-dir
ls -1 | grep -v ^$EXCLUDE | xargs -I{} mv {} $TARGET
mv * /foo
mv `ls` /foo
mv * /tmp/blah/
mv `ls` somewhere/
mv `ls *.boo` subdir
mv /mnt/usbdisk/[^l]* /home/user/stuff/.
mv /path/subfolder/* /path/
rsync -a --progress --remove-source-files src/test/ dest
find sourceDir -mindepth 1 -maxdepth 1 -exec mv --target-directory=destDir '{}' +
find sourceDir -mindepth 1 -maxdepth 1 -print0 | xargs -0 mv --target-directory=destDir
find sourceDir -mindepth 1 -exec mv "{}" --target-directory=destDir \;
find sourceDir -mindepth 1 -print0 | xargs -0 mv --target-directory=destDir
find /path/subfolder -maxdepth 1 -type f -name '*' -exec mv -n {} /path \;
find . -name some_pattern -print0 | xargs -0 -I % mv % target_location
mv /source/path/{.[!.],}* /destination/path
mv /path/subfolder/{.,}* /path/
cat $i | xargs mv -t dir.$count
find . -type f -iname '*.cpp' -exec mv -t ./test/ {} \+
find . ! -name . -prune -name '*.old' -exec mv {} ../old/ \;
mv ~/Linux/Old/^Tux.png ~/Linux/New/
grep -L -Z -r 'Subject: \[SPAM\]' . | xargs -0 -I{} mv {} DIR
grep -l 'Subject: \[SPAM\]' | xargs -I '{}' mv '{}' DIR
mv $(grep -l 'Subject: \[SPAM\]' | awk -F ':' '{print $1}') your_file
find sourceDir -print0 | xargs -0 mv -t destDir
find . -exec mv '{}' ~/play/ \;
find . | xargs -I'{}' mv '{}' ~/play/
mv /path/subfolder/.* /path/
mv wordpress/.* .
nl -n rz ca | awk -vOFS="\t" '/Ca/{$1="#"$2} {$1=$1}1' | sort -k1,1 | cut -f2-
find ./ -type f -print | xargs    -i mv -f {} ./newdir
find ./ -type f -print | xargs    -l56 -I {} mv -f {} ./newdir
find . -name "*.c" -print0 | xargs -0 -n1  -I '{}' mv '{}' temp
find "$path_to_folders" -name 'm?' -type d -exec mv {} {}.mbox \; -exec mkdir {}.mbox/Messages \; -exec sh -c "mv {}.mbox/*.emlx {}.mbox/Messages" \;
find . -name 'm?' -type d -exec mv '{}' '{}.mbox' ';' -exec mkdir '{}.mbox/Messages' ';' -exec sh -c 'mv {}.mbox/*.emlx {}.mbox/Messages' ';'
find /path/to/folders/* -type d -exec mv {} {}.mbox \; -exec mkdir {}.mbox/Messages \;
find /path/to/folders/* -type d  -exec mv {} {}.mbox \; -exec mkdir {}.mbox/Messages \; -exec sh -c "mv {}.mbox/*.emlx {}.mbox/Messages" \;
find $sourcePath -type f -mtime +10 -name "*.log" -exec mv {} $destPath \;
find . -atime +1 -type f -exec mv {} TMP \;
mv server.log logs/$(date -d "today" +"%Y%m%d%H%M").log
find ./ -maxdepth 1 -name "some-dir" -type d -print0 | xargs -0r mv -t x/
mv "$2" "`dirname $1`"
mv "/tmp/`basename $1`" "`dirname $2`"
find -maxdepth 1 -name '*.pdf' -exec rm "{}" \;
find . -maxdepth 1 -name "*.pdf" -print0 | xargs -0 rm
nl -nrz -w9  /etc/passwd
nl -nrz -w9 foobar
nl
nl -s- -ba -nrz
cat files | sort -t- -k2,2 -n
tr '.' ' ' | sort -nu -t ' ' -k 1 -k 2 -k 3 -k 4 | tr ' ' '.'
sort -nr bb
sort -n out
sort -n
sort -nk 2,2 file.dat | tac
sort -nrk 2,2 file.dat
tac files | sort -t- -k2,2 -n
sort -b -n -k2.4 table
tac temp.txt | sort -k2,2 -r -u
sort -n -k 2
sort -nrk 2,2
scp -3 user@server_b:/my_folder/my_file.xml user@server_b:/my_new_folder/
scp user@server_b:/my_folder/my_file.xml user@server_b:/my_new_folder/
sudo vim `which charm`
ssh -D1080 root@localhost -g
ssh user@host -M -S /tmp/%r@%h:%p -N
ssh user@host -S /tmp/%r@%h:%p
ssh user@host -X
find . -name "*.c" -print | vim -
find . -name '*.java' -exec vim {} +
find . -name '*.java' | xargs vim
find / -name filename -exec nano '{}' \;
info gcc "option index"
info -O gawk
info gcc --index-search=funroll-loops
info gcc "option index"
info bash 'Basic Shell Features' 'Shell Expansions' 'Filename Expansion' 'Pattern Matching'
tmux
basename "some/unknown/amount/of/sub/folder/file.txt"
basename /foo/bar/stuff
basename testFile.txt.1 .1
sed 's/^,/ ,/' test2.csv | tr -d \" | column -s, -t
diff file2 file1 | grep '^>' | sed 's/^>\ //'
uniq -w12 -c file
awk -F"\t" 'NF>1' file
join -1 2 -2 1 -t, BigFile.csv LittleFile.csv
join -t: selection2.txt selection1.txt
yes | sed -n '200000000,${=;p};200000005q'
awk 'NR==16224, NR==16482-1; NR==16482 {print; exit}' file
awk 'NR==16224, NR==16482' file
echo "foo.tar.gz" | rev | cut -d"." -f1 | rev
join -v1 success.txt fail.txt
echo $(basename "$1")
basename "some/unknown/amount/of/sub/folder/file.txt"
basename /EBF/DirectiveFiles/data_report_PD_import_script_ABF1_6
basename /path/to/dir/filename.txt .txt
nl -ba | sed 786q | grep . | awk '{print $2$1}'
yes
echo Hostname=$(hostname) LastChecked=$(date)
echo ${filename%.*}
yes -- "-tcp" | head -n 2
yes -- -tcp | head -n 2
shred -v -n 1 /path/to/your/file #overwriting with random data
shred -v -n 1 -z -u /path/to/your/file
shred -v -n 0 -z -u /path/to/your/file #overwriting with zeroes and remove the file
shred my-existing-file
shred -u $FILE
shred -uzn 35 filename
less -S file
od -xcb input_file_name | less
cat BIG_FILE.txt | less
less -p regex file_name
cat -vet file | less
cat -e yourFile | less
paste -d ' ' file <(rev file)
scp -vp me@server:/location/files\*
find . -exec echo {} +
find /ftp/dir/ -size +500k -iname "*.jpg"
find . -iname "*filename*"
find /var/www/vhosts/*/httpdocs -type f -iwholename “*/wp-includes/version.php” -exec grep -H “\$wp_version =” {} \;
find /home/*/public_html/ -type f -iwholename “*/wp-includes/version.php” -exec grep -H “\$wp_version =” {} \;
rename -n 's/special/regular/' **
rsync -nvraL test/a test/dest --exclude=a/b/c/d
find . -print0 | xargs -0
find downloads  -iname "*.gif"
find / -iname TeSt123.txt
find /tmp -name core -type f -print0 | xargs -0 /bin/rm -f
source <( sed 's/:\(.*\)/="\1"/' msgs.config )
cat ips | xargs -i% ping -c 2 %
cat ips | xargs -n1 ping -c 2
echo $(seq 254) | xargs -P255 -I% -d" " ping -W 1 -c 1 192.168.0.% | grep -E "[0-1].*?:"
ping -c 1 hostname | grep 192.168.11 | grep 'bytes from' | awk '{print $4}' | sed 's/://g'
ping -b 10.10.0.255
echo -e "\n\n\n" | ssh-keygen -t rsa
cat somedata.txt | "$outfile"
yes | awk 'FNR<4 {print >>"file"; close("file")}  1' | more
ls | read var
bg %  so it wont die when you logoff
popd
ls | xargs -I {} mv {} Unix_{}
ls | xargs -i mv {} unix_{}
find * -maxdepth 0 ! -path . -exec mv {} PRE_{} \;
ls | xargs -I {} mv {} PRE_{}
nl filename
ping google.com | xargs -L 1 -I '{}' date '+%+: {}'
history "$@" | tac | nl | tac | sed 's/^\( *\)\([0-9]\)/\1-\2/'
history "$@" | tac | nl | tac
ssh -n $R_USER@$R_HOST 'touch /home/user/file_name.txt'
echo "Cannot acquire lock - already locked by $(cat "$lockfile")"
comm -13 <(grep '#include' file1 | sort) <(grep '#include' file2 | sort)
nl | sort -R | cut -f2 | head -"${1:-10}"
echo $PATH | awk -F: -f rem_dup.awk | paste -sd:
echo -n $line | od -x
find "$somedir" -maxdepth 0 -empty -exec echo {} is empty. \;
echo -n *Checking Partition Permission* Hostname=$(hostname) LastChecked=$(date)
tail -n +11 /tmp/myfile
echo I am $(whoami) and the program named ls is in $(which ls).
echo " RDBMS exit code : $RC  "     | tee -a ${LOG_FILE}
echo "This is   a sentence." | tr -s " " "\012"
echo "Total generated: $(echo "$generated_ports" | sort | uniq | wc -l)."
echo -e "a\nb\ncccccccccccc\nd" | paste - - | column -t
echo "deb http://ppa.launchpad.net/webupd8team/java/ubuntu precise main" | tee -a /etc/apt/sources.list
echo "deb-src http://ppa.launchpad.net/webupd8team/java/ubuntu precise main" | tee -a /etc/apt/sources.list
cat ips | xargs -n1 echo ping -c 2
cat ips | xargs echo ping -c 2
tail -n +1 file1.txt file2.txt file3.txt
echo hello `whoami`
tail -n +1000001 huge-file.log
find /some/dir/ -maxdepth 0 -empty -exec echo "huzzah" \;
cat new.txt  | nl
cat new.txt  | nl | sed  "/2/d"
cat new.txt  |  nl |sed  "3d;4d"
echo "on" | tee /sys/bus/usb/devices/usb*/power/level
echo "hello world" | echo test=$(cat)
comm -2 file1.txt file2.txt | awk -F'\t' '{print (NF==2?"y":"")}'
yes yes | sed -e 5s/yes/no/ -e 5q
find "$d" -prune -empty -type d
find "$d" -prune -empty
echo -e "HTTP/1.1 200 OK\n\n $(date)"
find -name file -exec echo '-exec is an action so an implicit -print is not applied' \;
find -name file -ok echo '-ok is an action so an implicit -print is not applied' \;
find -name file -okdir echo '-okdir is an action so an implicit -print is not applied' \;
echo -en '111 22 3\n4 555 66\n' | tr ' ' '\t'
find -name file -printf 'Since -printf is an action the implicit -print is not applied\n'
find -name file -exec echo 'This should print the filename twice if an implicit -print is applied: ' {} +
ls -alFt `find . -name "bla.txt"` | rev | cut -d" " -f1 | rev | head -1
find ./C -name "*.c" | xargs -n1  echo cp xyz.c
echo 'hello, world' | cat
cat file | fold -w29
cat file | fold -s -w29
cat file | xargs -n3
find /some/dir/ -maxdepth 0 -empty -exec echo "huzzah" \;
cat -n infile
find /some/dir/ -maxdepth 0 -empty
od -A n -t d -N 1 /dev/urandom
echo {1..9}: 10 | tr -d ' '
yes '#' | head -n 10 | tr -d '\n'
yes x | head -n 10
yes x | head -10 | awk 'BEGIN { RS = "%%%%%%%" } { split($0,a,"\n"); for (i=1; i<length(a); i+=4) print a[i], a[i+1], a[i+2], a[i+3] }'
head -c 1000 /dev/zero | tr '\0' '*'
yes 123456789 | head -2
yes | head -3
echo $(yes % | head -n3)
yes ' ' | head -7 | tr -d '\n'
dig @ns1.newnameserver domain. a
dig @ns2.newnameserver domain. a
dig @some.other.ip.address domain. a
dig @8.8.8.8 domain. a
who | awk -F '[()]' '/orschiro/{print $(NF-1)}' | grep -v orschiro | uniq
who | sed -e '/orschiro/! d; /pts/! d; s/^.*\(:[0-9.]\+\).*$/\1/p;d' | head -n1
hostname -I|cut -d" " -f 1
hostname  -i
hostname --ip-address
hostname -I
hostname -I | awk '{print $1}'
hostname -I | cut -d' ' -f1
dig @some.other.ip.address domain. ns
dig @8.8.8.8 domain. ns
dig @server hostname.bind ch txt
yes 'http://www.blabla.bla/forum-detail/?ft=72260&fid=34&&pgr=' | nl -ba | sed 786q | grep . | awk '{print $2$1}'
echo "$1" | od -xcb
find . -type d -print0
find $HOME -maxdepth 1 -type f -name '.*' -print0
find $root -type d -printf '%p:'
find $root -type d | tr '\n' ':'
find "$root" -name ".[a-z]*" -prune -o -type d -printf '%p:'
find ~/code -type d | tr '\n' ':' | sed 's/:$//'
find ~/code -type d -name '[^\.]*' | tr '\n' ':' | sed 's/:$//'
find ~/code -name '.*' -prune -o -type d -printf ':%p'
find ~/code -name '.*' -prune -o -type f -a -perm /u+x -printf ':%h\n' | sort | uniq | tr -d '\n'
find ~/code -type d | sed '/\/\\./d' | tr '\n' ':' | sed 's/:$//'
sort ips.txt | uniq -c | sort -bgr
sort ports.txt | uniq -c | sort -r
ls -1 | tr '[A-Z]' '[a-z]' | sort | uniq -c | grep -v " 1 "
sort filename | uniq -c
sort filename | uniq -c | sort -nr
sort | uniq -c
sort | uniq -c | sort -n
sort ip_addresses | uniq -c
sort -n ip_addresses.txt | uniq -c
tree | tail -1
find . -type f -ls
awk -F '\t' '{print $2}' * | sort | uniq -c | sort -nr
echo Aa | od -t x1
echo "$DIREC" | od -c
seq  -f "#" -s '' 10
cat /dev/urandom | tr -dc '. ' | fold -w 100 | head -1
seq -s % 4|tr -d '[:digit:]'
seq -s= 100|tr -d '[:digit:]'
find . -name “*.jpg”
find . -name *.code
find . | sort -f | uniq -i -d
find . -type f | awk -F/ '{print $NF}' | sort -f | uniq -i -d
find . |sed 's,\(.*\)/\(.*\)$,\1/\2\t\1/\L\2,'|sort|uniq -D -f 1|cut -f 1
find /var/log/
find . -maxdepth 1 -type f -print0
find / -newerct '1 minute ago' -print
ls | sort -f | uniq -i -d
diff -q /dir1 /dir2|cut -f2 -d' '
tree -fi |grep -v \>
find sort_test/ -type f | env -i LC_COLLATE=C sort
find sort_test -type f | env -i LC_COLLATE=en_US.UTF-8 sort
find sort_test/ -type f | env -i LC_COLLATE=en_US.utf8 sort
find -L. -type l
who | cut -d' ' -f1 | sort | uniq
find /usr/local/etc/rc.d -type f | awk -F/ '{print $NF}'
$ find other -maxdepth 1
find other -maxdepth 1 -printf "%P\n"
diff -dbU0 a b
diff -dbU0 a b | tail -n +4 | grep ^- | cut -c2-
diff -burNad teste1 teste2
ping google.com | awk '{ sent=NR-1; received+=/^.*(time=.+ ms).*$/; loss=0; } { if (sent>0) loss=100-((received/sent)*100) } { print $0; printf "sent:%d received:%d loss:%d%%\n", sent, received, loss; }'
seq 2000 65000 | sort -R | head -n 1
seq 1 10 | sort -R | tee /tmp/lst |cat <(cat /tmp/lst) <(echo '-------') | tac
seq 1 10 | sort -R | tee /tmp/lst |cat <(cat /tmp/lst) <(echo '-------')
yes | head -n 10 | awk '{printf( "%03d ", NR )}'
yes | head -n 10 | awk '{printf( "%03d ", NR )}'    ##for 01..10
yes | head -n 100 | awk '{printf( "%03d ", NR )}'
yes | head -n 100 | awk '{printf( "%03d ", NR )}'   ##for 001...100
find bla -name *.so -print0 | sort -rz
find -name '*.jpg' | sort -n
find ~/Music/ -iname 'cover.*' -printf '%h\n' | sort -u
find /folder/of/stuff -type f | sort
find . -type f -name "*.*" | awk -F. '{print $NF}' | sort -u
find . -type f | grep -o -E '\.[^\.]+$' | sort -u
find . -type f | sed -e 's/.*\.//' | sed -e 's/.*\///' | sort -u
find . -type f -name "*.???" | awk -F. '{print $NF}' | sort -u
find . -type f | sed -e 's/.*\.//' | sed -e 's/.*\///' | sort | uniq -c | sort -rn
find ~/Music/ -maxdepth 2 -mindepth 2 -type d | sort
find / -name '<name_pattern>' -type d | sort | uniq
seq 10 | xargs
seq 10 | xargs echo -n
find --help
sed -e 's/\t/_|/g' table.txt |  column -t -s '_' | awk '1;!(NR%1){print "-----------------------------------------------------------------------";}'
ping host | awk '{if($0 ~ /bytes from/){print strftime()"|"$0}else print}'
od -cvAnone -w1 | sort -b | uniq -c | sort -rn | head -n 20
od -cvAnone -w1 | sort -bu
echo "Welcome $(whoami)!"
echo "Welcome `whoami`!"
echo -ne "Welcome $(whoami)!\n"
echo -ne "Welcome `whoami`!\n"
readlink -f PATH
readlink -f YOUR_PATH
readlink -f $(which java)
grep -Eo '([0-9]+-){3}[0-9]+' infile | tr - .
cal -h | cut -c 4-17 | tail -n +3
find $root -type d -printf '%p:'
find "$root" -name ".[a-z]*" -prune -o -type d -printf '%p:'
echo "He likes cats, really?" | fold -w1 | sort -u
find /proc -print0 | xargs -0
find /proc | xargs
find /usr/src -not \( -name "*,v" -o -name ".*,v" \) '{}' \; -print
find /proc -exec ls '{}' \;
find /proc -print0 | xargs -0
find /proc | xargs
find . -name SCCS -prune -o -print
find . -print -name SCCS -prune
comm -12 <(grep -rl word1 . | sort) <(grep -rl word2 . | sort)
ls -1 | paste -sd "," -
find . -type f -print0 | tr '\0' ','
find . -type f | paste -d, -s
find / -group name_of_group
find / -size +1000 -mtime +30 -exec ls -l {} \;
find / -type f -exec echo {} - ';' -o -exec echo {} + ';'
find ... -print0
find $WHATEVER -printf "%s %p\n"
cat report.txt | grep -i error
tac file | awk '/pattern/{print;exit}1' | tac
tac file | sed '/pattern/q' | tac
sed -n '/pattern/!p' file
seq 10 | tac | sed '1,3d' | tac
who | awk '{print "The user " $1 " is on " $2}'
which -a python
find . -not -path '*/\.*'
grep "$(cat file1.txt)" file2.txt
find | xargs
comm -23 <(find dir1 -type d | sed 's/dir1/\//'| sort) <(find dir2 -type d | sed 's/dir2/\//'| sort) | sed 's/^\//dir1/'
comm -23 <(find dir1 -type f | sed 's/dir1/\//'| sort) <(find dir2 -type f | sed 's/dir2/\//'| sort) | sed 's/^\//dir1/'
comm -23 <(find dir1 | sed 's/dir1/\//'| sort) <(find dir2 | sed 's/dir2/\//'| sort) | sed 's/^\//dir1/'
cat $1.tmp | sort -u
who | awk '{ print $1, $2 }'
who | cut -d " " -f1,2
df -Ph $PWD | tail -1 | awk '{ print $3}'
df . | awk '$3 ~ /[0-9]+/ { print $4 }'
df . -B MB | tail -1 | awk {'print $4'} | cut -d'%' -f1
df . -B MB | tail -1 | awk {'print $4'} | grep  .[0-9]*
df $PWD | awk '/[0-9]%/{print $(NF-2)}'
find /usr/ports/ -name work -type d -print -exec rm -rf {} \;
find -mindepth 1 -maxdepth 1 -type d | cut -c 3- | sort -k1n | tail -n 1 | xargs -r echo rm -r
ping -c 25 google.com | tee >(split -d -b 100000 - /home/user/myLogFile.log)
tail -f /var/log/some.log | grep --line-buffered foo | grep bar
tail -f /var/log/syslog
tail -f file | grep --line-buffered my_pattern
yes $1 | head -$number
find /home/kibab -name file.ext -exec echo . ';'
basename "$FILE" | cut -d'.' -f-1
set | grep ^fields=\\\|^var=
paste <(cal 2 2009) <(cal 3 2009) <(cal 4 2009)
readlink -f  /path/here/..
readlink -m /path/there/../../
echo 'abcdefg'|tail -c +2|head -c 3
echo "$(comm -12 <(echo "$a" | fold -w1 | sort | uniq) <(echo "$b" | fold -w1 | sort | uniq) | tr -d '\n')"
find . -type f -exec echo chmod u=rw,g=r,o= '{}' \;
seq $(tail -1 file)|diff - file|grep -Po '.*(?=d)'
history
cat /proc/17709/cmdline | xargs -0 echo
ps | egrep 11383 | tr -s ' ' | cut -d ' ' -f 4
comm -12 <(echo $a|awk -F"\0" '{for (i=1; i<=NF; i++) print $i}') <(echo $b|awk -F"\0" '{for (i=1; i<=NF; i++) print $i}')|tr -d '\n'
comm -12  <(ls 1) <(ls 2)
comm -12  <(ls one) <(ls two)
comm -12 file1 file2
comm -12 <(sort set1) <(sort set2)
comm -12 ignore.txt input.txt
comm -12 <(comm -12 <(comm -12 <(sort file1) <(sort file2)) <(sort file3)) <(sort file4)
cat `find . -name '*.foo' -print`
$ cat 1
cat -vet a
rev domains.txt | cut -d '.' -f 2- | rev
rev file
cat -v -e filename
cat whatever | egrep 'snozzberries|$'
cat /etc/passwd /etc/group
find . -name '*.foo' -exec cat {} \;
cat $(find . -name '*.foo')
find [whatever] -exec cat {} +
find [whatever] -exec cat {} \;
find [whatever] -print0 | xargs -0 cat
find [whatever] | xargs cat
find . -type f -exec cat {} \; -print
od -t x1 -An file |tr -d '\n '
grep -ao "[/\\]" /dev/urandom|tr -d \\n
cat /dev/urandom | tr -dc '. ' | fold -w 100
cat list_part* | sort --unique | wc -l
cat /etc/passwd | sed 's/^\([^:]*\):.*$/crontab -u \1 -l 2>\&1/' | sh | grep -v "no crontab for"
date -u -Iseconds
date +%s
ps  -ef | grep $$ | grep -v grep
find ./work/ -type f -name "*.sh" -mtime -20 | xargs -r ls -l
diff -bur folder1/ folder2/
diff "${@:3}" <(sort "$1") <(sort "$2")
tree -d -L 1 -i --noreport
mount | awk '$3 == "/pa/th" {print $1}'
echo -n "Hello" | od -A n -t x1
echo orange | fold -w 1
echo "abcdefg" | fold -w1
paste <(paste -d" " f1 f2) f3
paste file file2 file3 | sed 's/\t/ /'
paste -d'¤' file1 file2 | sed 's,¤, ,g'
paste -d" " file1 file2 | paste -d'|' - file3 | sed 's,|, ,g'
join -v 1 <(sort file1) <(sort file2)
sort file1.txt file2.txt file2.txt | uniq -u
sort set1 set2 | uniq
paste tmp/sample-XXXX.{tim,log}
sort file1 file2 | uniq -u
finger -l | grep "Name:" | cut -d ":" -f 3 | cut -c 2- | sort | uniq
finger | awk 'NR>1{print $2,$3}'
finger -l | grep "Name:" | tr -s ' ' | cut -d " " -f 2,4- | sort | uniq
cat ip_addresses | sort | uniq -c | sort -nr | awk '{print $2 " " $1}'
sort file1 file2 | uniq -d
yes $'one\ntwo' | head -10 | nl | sort -R | cut -f2- | head -3
set | grep ^IFS=
date -d "Oct 21 1973" +%s
comm -12 <(zcat number.txt.gz) <(zcat xxx.txt.gz)
fold -b -w 20 | cut --output-delimiter $'\t' -b 1-3,4-10,11-20
fold -w3
yes | cat -n | head -10 | awk 'NR % 4 == 1'
find -printf "%y %i %prn"
find /path/to/files -type f -name \*.cfg  -print -exec cat {} \; -exec sleep 2 \;
paste -sd',,\n' file
cat file | paste -d' ' - -
cat file | paste -d\ - - -
paste -d' ' <(sed -n 'p;n' num.txt) <(sed -n 'n;p' num.txt)
find . -name 'my*' -type f -ls
echo "$FILE" | cut -d'.' -f2
echo $(ls -l $(which bash))
ls -l `which passwd`
ls -l "$( which studio )"
which studio | xargs ls -l
ls `which g++` -al
ls `which gcc` -al
echo "$FILE" | cut -d'.' -f1
echo "$FILE" | cut -d'.' --complement -f2-
echo "$FILE"|rev|cut -d"." -f3-|rev
ls /home/ABC/files/*.csv | rev | cut -d/ -f1 | rev
tree -Csu
find . -name '*.ear' -exec du -h {} \;
df
df -H --total /
df -k .
df -h /
df -Ph | column -t
df .
df --total
a=$( df -H )
cat `which java` | file -
file -L `which gcc`
file `which c++`
file `which file`
which file | file -f -
file $(which foo)
file `which python`
find /directory -newermt $(date +%Y-%m-%d -d '1 day ago') -type f -print
echo -en "${line:0:11}" "\t" $(md5sum "${line:12}") "\0"
paste -d':' *.txt | sed 's/ [^:]*$//;s/ [^:]*:*/ /g;s/://g'
echo $string | cut -d';' -f1
echo "<line>" | cut -d ";" -f 1
grep -o '^\S\+' <(comm file1 file2)
cat -v /dev/urandom
cat text.txt | tr -s ' ' | cut -d ' ' -f4
cat text.txt | cut -d " " -f 4
echo `date -v-1d +%F`
which c++
which gradle
which programname
which python
which python2.7
groups $(who | cut -d' ' -f 1)
cat --help
history | awk '{sub($1, ""); sub(/^[ \t]+/, ""); print}'
history | awk '{sub($1, "", $0); sub(/^[ \t]+/, "", $0); print}'
echo "$(hostname):$(cat /sys/block/sda/size)"
echo -n `hostname`
find */ | cut -d/ -f1 | uniq -c
df
df --total
cat /proc/1/sched  | head -n 1
who -la
ps -ef | grep $0 | grep $(whoami)
mount -v | grep " on / "
tree -afispugD --inodes | awk '{FS="./"; ORS=""; printf("%-60s%s\n",$NF,$0)}'
echo 'your, text, here' | fold -sw 70
cat file | xargs
history 10
history 10 | cut -c 8-
cal 4 2009 | tr ' ' '\n' | grep -v ^$ | tail -n 1
echo 0a.00.1 usb controller some text device 4dc9 | rev | cut -b1-4 | rev
cat $filename | sed "${linenum}p;d";
cat /etc/passwd -n | grep `whoami` | cut -f1
wc `find`
wc `find | grep .php$`
seq 1 100000 | sed -n '10000,10010p'
seq 1 100000 | tail -n +10000 | head -n 10
cat dump.txt | head -16224 | tail -258
cat file | head -n 16482 | tail -n 258
history | sed -n '2960,2966p'
tail -n +347340107 filename | head -n 100
grep -e TEXT *.log | cut -d':' --complement -s -f1
join -t " " -j 1 <(sort file1) <(sort file2)
sort <(sort -u file1.txt) file2.txt file2.txt | uniq -u
sort foo.txt | uniq
grep -w -v -f blacklist file
grep -v 'pattern' filename
comm -23 <(sort a.txt) <(sort b.txt)
comm -23 <(sort file1) <(sort file2)
comm -13 <(sort file1) <(sort file2)
comm -13 <(sort first.txt) <(sort second.txt)
tac file | rev
cat <(grep -vxF -f set1 set2) <(grep -vxF -f set2 set1)
comm file1 file2
gcc -print-search-dirs | sed '/^lib/b 1;d;:1;s,/[^/.][^/]*/\.\./,/,;t 1;s,:[^=]*=,:;,;s,;,;  ,g' | tr \; \\012
groups | tr \  \\n
who | awk '{ print $1 }'
who | sed -e 's/[ \t].*//g'
mount | awk '/\/dev\/sd/ {print NR, $1, $3}'
df -h |  awk '{print $1}'
diff -q /dir1 /dir2 | grep /dir1 | grep -E "^Only in*" | sed -n 's/[^:]*: //p'
dig @"127.0.0.1"  _etcd-client._tcp. SRV
find . ! -local -prune -o -print
who | awk '{ if (NR!=1 && NR!=2) {print} }' | sed -e 's/ /, /g'
finger -l | awk '/^Login/'
finger -l | awk '/^Login/' | sed 's/of group.*//g'
find . -type f ! -size 0 -exec ls -l '{}' \;
df "$filename" | awk 'NR==1 {next} {print $6; exit}'
df "$path" | tail -1 | awk '{ print $6 }'
df -P "$path"  | tail -1 | awk '{ print $NF}'
df -P "/tmp" | awk 'BEGIN {FS="[ ]*[0-9]+%?[ ]+"}; NR==2 {print $NF}'
echo -e "ONBOOT=\"YES\"\nIPADDR=10.42.84.168\nPREFIX=24" | sudo tee -a /etc/sysconfig/network-scripts/ifcfg-eth4
df -P "$path" | awk 'BEGIN {FS="[ ]*[0-9]+%?[ ]+"}; NR==2 {print $1}'
df -h $path | cut -f 1 -d " " | tail -1
yes | nl -ba | tr ' ' 0 | sed 100q | cut -b 4-6
wc `find . -name '*.[h|c|cpp|php|cc]'`
find . -name \*.java | tr '\n' '\0' | xargs -0 wc
echo MYVAR | grep -oE '/[^/]+:' | cut -c2- | rev | cut -c2- | rev
comm -2 -3 <(sort -n B.txt) <(sort -n B.txt)
cat $file | wc -c
comm -12 <(sort -u /home/xyz/a.csv1) <(sort -u /home/abc/tempfile) | wc -l
ls -d -1 $PWD/**/*/* | cat -n
ls | grep "android" | cat -n
yes | head -n10 | grep -n . | cut -d: -f1 | paste -sd:
seq 10 | xargs -P4 -I'{}' echo '{}'
seq 10 | awk 'NR%2{printf("%s ", $0); next}1'
seq 10 | paste -sd" \n" -
seq 10 | sed '2~2G' | awk -v RS='' '{$1=$1; print}'
seq 10 | sed 'N;s/\n/ /'
seq -w 1 10
seq 1 100
seq -f "%02g" 30
seq -w 30
seq -w 30
seq 5 | awk '{printf "%s", $0} END {print ""}'
seq 5 | awk '{printf "%s", $0}'
seq $1
sort -n ip_addresses.txt | uniq -c
echo "$PATH" | rev | cut -d"/" -f1 | rev
cat /dev/urandom | tr -dc 'a-zA-Z0-9'
comm -1 -2 <(ls /dir1 | sort) <(ls /dir2 | sort)
comm -1 -2 file1.sorted file2.sorted
comm -1 -2 <(sort file1) <(sort file2)
echo $name | tr -c -d 0-9
cat file | fold -w29 | head -1
cat file | fold -s -w29 | head -1
cut -d: -f1 /etc/group
cat file1.txt | grep -Fvf file2.txt | grep '^Q'
finger -s | awk '{printf("%s %s\n", $1, $2);}'
cat /dev/urandom | tr -dC '[:graph:]'
cat datafile | rev | cut -d '/' -f 2 | rev
comm -1 -3 file1 file2
find foo/// bar/// -name foo -o -name 'bar?*'
wc -l $f | tr -s ' ' | cut -d ' ' -f 1
cat set1 set2 | sort -u
cat -n file_name | sort -uk2 | sort -nk1 | cut -f2-
find * -type f  | xargs md5sum | sort | uniq -Dw32 | awk -F'[ /]' '{ print $NF }' | sort -f | uniq -Di
set -x
find /home/folder1/*.txt -type f | awk -F '.txt' '{printf "ln -s %s %s_CUSTOM_TEXT.txt\n", $0, $1}'
find ~ -name '*.txt' -print0 | xargs -0 cat
find ~/ -name '*.txt' -exec cat {} ;
readlink -ev mypathname
find . -printf "%y %p\n"
find mydir -type d
~ $ . trap.sh | cat
echo <(yes)
find . -name SCCS -prune -o -print
find . -print -name SCCS -prune
du | awk '{print $2}'
df -k $FILESYSTEM | tail -1 | awk '{print $5}'
df . -B MB | tail -1 | awk {'print substr($5, 1, length($5)-1)'}
tree -p -d
cal 09 2009 | awk 'BEGIN{day="9"}; NR==4 {col=index($0,day); print col }'
cal 09 2009 | awk 'NR==4{day="9"; col=index($0,day); print col }'
cal 09 2009 | awk '{day="9"; if (NR==4) {col=index($0,day); print col } }'
pstree | cat
tee
readlink /dev/disk/by-uuid/b928a862-6b3c-45a8-82fe-8f1db2863be3
dig -x 72.51.34.34
dig -x 127.0.0.1
j=`echo $i | rev | cut -d "." -f2`;
yes '' | nl -ba | sed -n -e 11,24p -e 24q
echo $string | cut -d';' -f2
echo "<line>" | cut -d ";" -f 2
tr -s ' ' | cut -d ' ' -f 2
echo -e "<line>" | sed 's:\s\+:\t:g' | cut -f2
cut -d ' ' -f 2
cut -d\  -f 2
cut "-d " -f2 a
find /myprojects -type f -name '*.cpp' -print0 |    xargs -0 echo sed -i 's/previousword/newword/g'
dig +short -f list
dig TXT +short o-o.myaddr.l.google.com @8.8.8.8
dig TXT +short o-o.myaddr.l.google.com @ns1.google.com
sort ip_addresses | uniq -c
who | awk '{ print $1; }' | sort -u | awk '{print $1; u++} END{ print "users: " u}'
dirname "$(readlink -f "$0")"
df -P $path | tail -1 | cut -d' ' -f 1
df . | tail -1 | awk '{print $1}'
echo `seq $start $end`
seq -s' ' $start $end
seq 10 | xargs echo
uniq
echo 123 | tee >(tr 1 a)  | tr 1 b
diff -rq dir1 dir2 | grep 'Only in dir1/'
diff -rq /path/to/folder1 /path/to/folder2
diff  --brief --recursive dir1/ dir2/
diff -q dir1 dir2
diff -qr dir_one dir_two | sort
diff -rqyl folder1 folder2 --exclude=node_modules
diff -arq folder1 folder2
echo $(basename $(readlink -nf $0))
head -$N file | tail -1 | tr ',' '\n'
find /proc -type d | egrep -v '/proc/[0-9]*($|/)' | less
history | awk '{print $2}' | awk 'BEGIN {FS="|"}{print $1}' | sort | uniq -c | sort -n | tail | sort -nr
cut -d' ' -f5 file | paste -d',' -s
find ~/bin/FilesDvorak/.* -maxdepth 0 | awk -F"/" '{ print $6 }'
who am i|awk '{ print $5}'
hostname  -I
hostname  -I | awk -F" " '{print $1}'
hostname  -I | cut -f1 -d' '
hostname -I | awk '{print $1}'
hostname -i
hostname -I
md5sum /path/to/destination/file
ps -u $(whoami) | grep firefox | awk '{printf $1}'
readlink -f "$path"
ls -d -1 $PWD/**/*/* | nl
ping -q -c 5 google.com | tail -n 1 | cut -f 5 -d '/'
ping -c 5 google.com | grep "round-trip" | cut -f 5 -d "/"
ping -c 4 www.stackoverflow.com | awk -F '/' 'END {print $5}'
ping -c 4 www.stackoverflow.com | sed '$!d;s|.*/\([0-9.]*\)/.*|\1|'
ping -c 4 www.stackoverflow.com | tail -1| awk '{print $4}' | cut -d '/' -f 2
basename "`pwd`"
basename $(pwd)
basename `pwd`
echo "$(basename $(pwd))"
pwd | xargs basename
pwd | grep -o '[^/]*$'
basename $(echo "a:b:c:d:e" | tr ':' '/')
who -b | awk '{$1=""; $2=""; print $0}' | date -f -
find . -type f | xargs | wc -c
cal 02 1956
seq 65 90 | awk '{printf("%c",$1)}'
echo "$b" | grep -o "[$a]" | tr -d '\n'
echo "$b" | grep --only-matching "[$a]" | xargs | tr --delete ' '
zcat "$file" | awk '{print NF}' | head
find ./ -name *.ogv -exec echo myfile {} \;
comm -12 <(awk '{print $3}' file1 | sort -u) <(awk '{print $3}' file2 | sort -u)
gunzip -l file.zip
sed 's/\n//' file
nl -b a file | sort -k1,1nr | sed '1, 4 d' | sort -k1,1n | sed 's/^ *[0-9]*\t//'
tail -n +2 "$FILE"
cat ${SPOOL_FILE}                   | tee -a ${LOG_FILE}
cat Little_Commas.TXT
nl file | sort -nr | cut -b8-
echo `sed -e 's/$/\ |\ /g' file`
cat filename
tail -n +2 foo.txt
cat my_script.py
cat n
cat numbers.txt
cat order.txt
cat xx.sh
cat ~/.ssh/config
echo "$(date): " $line
echo $(date) doing stuff
echo "$(date +%H:%M:%S): done waiting. both jobs terminated on their own or via timeout; resuming script"
echo "The current default java is $(readlink --canonicalize `which java`)"
find -maxdepth 0
find -mindepth 0 -maxdepth 0
find -prune
echo "$PWD" | sed 's!.*/!!'
tree
tree -p
tree -s
tree -D
ps | tail -n 4 | sed -E '2,$d;s/.* (.*)/\1/'
who -m | awk '{print $1;}'
cat /var/spool/mail/`whoami`
echo "$(pwd)/$(basename "$1")"
echo pwd: `pwd`
pwd -P
echo -n $(pwd)
echo `date` `hostname`
echo `date +"%a %x %X"` `hostname`
date --date='1 days ago' '+%a'
date -d "$(date -d "2 months" +%Y-%m-1) -1 day" +%a
diff <(fold -w1 <(sed '2q;d' $f)) <(fold -w1 <(sed '3q;d' $f)) | awk '/[<>]/{printf $2}'
find $SrvDir* -maxdepth 0 -type d
echo "dirname/readlink: $(dirname $(readlink -f $0))"
dirname `pwd -P`
dirname `readlink -e relative/path/to/file`
echo $(dirname $(readlink -m $BASH_SOURCE))
mount | tail -1 | sed 's/^.* on \(.*\) ([^)]*)$/\1/'
find -empty
find empty1 empty2 not_empty -prune -empty
history
cat "text1;text2;text3" | sed -e 's/ /\n/g'
cat `find [whatever]`
cat "$(which f)"
cat `which f`
find . -type f -printf "%f %s\n"
find full_path_to_your_directory -type f -printf '%p %s\n'
find . -iname "*.txt" -exec du -b {} +
find . -name "*.txt" -print0 |xargs -0 du -ch
df -P file/goes/here | tail -1 | cut -d' ' -f 1
df | grep /dev/disk0s2
find * -maxdepth 0 -type d -print0
ls -1 | tr '\n' ','
ls -1 | tr '\n' ',' | sed 's/,$/\n/'
ls | sed '$!s/$/,/' | tr -d '\n'
ls -1b | tr '\n' ';'
ls -m | tr -d ' ' | tr ',' ';'
find . -type l -print | xargs ls -ld | awk '{print $10}'
find . -type f -exec echo {} {} \;
find /tmp  | head
find | head
tree --help |& head -n2
cat /dev/urandom | LC_ALL=C tr -dc 'a-zA-Z0-9' | fold -w 24 | head -n 1
cat /dev/urandom | tr -cd 'a-f0-9' | head -c 32
cat /dev/urandom | env LC_CTYPE=C tr -cd 'a-f0-9' | head -c 32
sed 's/\(.....\)\(.....\)/\1\n\2/' input_file | split -l 2000000 - out-prefix-
zcat "$line" | head -n5
awk -F, '{ if (NR == 1)print}{if($3 == "f")print}' input | column -t -s,
head -n 1 filename | od -c
seq 1 10000 | head -1
find $HOME/. -name *.txt -exec head -n 1 -v {} \;
find xargstest/ -name 'file?B' | sort | xargs head -n1
head -1 <(sort set)
od --read-bytes=2 my_driver
sed 's/\([^ ]*\) /\1\n/' input | fold -w 100
finger $USER |head -n1 |cut -d : -f3
find . -name "file.ext" -execdir pwd ';'
ls "`pwd`/file.txt"
xargs -n 1 -I '{}' find "$(pwd)" -type f -inum '{}' -print
which cc
which gcc
which rails
which lshw
tree -fi
echo $(readlink -f /dev/disk/by-uuid/$1) is mounted
echo $(readlink -f /dev/disk/by-uuid/$1) is not mounted
echo "$NAME" | cut -d'.' -f2-
cat files.txt | xargs du -c | tail -1
df --total -BT | tail -n 1
tree --help
split --help
echo Hello world | od -t x1 -t c
finger $username | awk '/^Directory/ {print $2}'
hostname
ping -c 2 -n 127.0.0.1 | awk -F'[ =]' -v OFS='\t' 'NR>1 { print $6, $10 }'
echo hello world | tee  >(awk '{print $2, $1}')
echo hello world | tee /dev/tty | awk '{print $2, $1}'
cat /proc/config.gz | gunzip
find . -type f -printf '%k %p\n' |sort -n |tail -n 20
find . -type f -printf '%s %p\n' | sort -rn | head -20
history | tail
history | tail -10
history | tail -n 10
tail great-big-file.log
tail -f /var/log/syslog
tail /var/log/syslog
tail -n 1000 /var/spool/cron/*
history | tail -1 | awk '{print $1}'
sed 's/^/./' | rev | cut -d. -f1  | rev
tail -1 $file1 | tee -a $file2
tail -1 <(sort set)
mount | tail -1 | sed 's/ on .* ([^)]*)$//'
echo "Your string here"| tr ' ' '\n' | tail -n1
echo "a b c d e" | tr ' ' '\n' | tail -1
awk '{print $NF}' file.txt | paste -sd, | sed 's/,/, /g'
nl -b a "<filename>" | grep "<phrase>" | awk '{ print $1 }'
uniq -c | sort -n | tail -n1
cat table.txt | awk '{print $1}' | sort | uniq  | xargs -i grep {} table.txt
join -v 2 index <(nl strings)
join <(sort index) <(nl strings | sort -b)
find . -name '*.txt' -print0|xargs -0 -n 1 echo
find /fss/fin -d 1 -type d -name "*" -print
find /myfiles -type d
find ./ -type d -print
find . \! -name BBB -print
find . -not \( -name .svn -prune -o -name .git -prune -o -name CVS -prune \) -type f -print0 | xargs -0 file -n | grep -v binary | cut -d ":" -f1
find .
find . -type f -exec grep -il confirm {} \;
find . -type f
find / -type f -exec echo {} \;
find ./ -type f -print
find . -maxdepth 1 -mindepth 1 -type d
find . -type d -exec ls -ld {} \;
find /mnt/raid -type d
find /etc   ! -name /etc
find /etc/. ! -name . -prune
find /etc/. ! -name /etc/.
find . ! -name . -prune
find . \( -name . -o -prune \)
find / -newerct '1 minute ago' -print
find "/zu/durchsuchender/Ordner" -name "beispieldatei*" -print0 | xargs -0 grep -l "Beispielinhalt"
find /tmp/a1
find . | egrep -v "(exclude3|exclude4)" | sort
find . -type f -not -path '*/\.*'
find . -path './.git' -prune -o -type f
find . -name .svn -a -type d -prune -o -print
find . -path '*/.svn*' -prune -o -print
find . -type d -name .svn -prune -o -print
find . -exec echo xx{}xx \;
find ~ -name 'Foto*'
find .  -mtime -14 | sed -e 's/^\.\///'
find -type d -maxdepth 1 ! -name ".*" -printf "%f\n"
find . -maxdepth 1 -type f -newermt "Nov 22" \! -newermt "Nov 23" -exec echo {} +
find . -type f -mtime -2 -exec echo {} +
find . -type f
find -maxdepth 1 -type d
find . -maxdepth 1 -mindepth 1 -type d -printf '%f\n'
find . -type d -maxdepth 1
find /path/to/dir/ -mindepth 1 -maxdepth 1 -type d
find . -mindepth 1 -maxdepth 1 -type d -printf "%P\n"
who | awk '{print $3 " " $4 " "$1}' | sort | head -1
ping google.com -n 10 | awk '/Minimum =/ { sub(",","",$3); print $3 }'
ping google.com -n 10 | grep Minimum | awk '{print $3}' | sed s/,//
ls -1tr * | tail -1
grep -Ff list1.txt list2.txt | sort | uniq -c | sort -n | tail -n1
mount | tail -1 | sed -e "s/^[^/]*\(.*\) type .*$/\1/g"
find file1 -prune -newer file2
find . -type f -printf "%f %s\n"
find dir -type f -printf "f %s %p\n"
find tmp -type f -printf "%s %p\n" | awk '{sub(/^[^ ]+/,sprintf("f %10d",$1))}1'
find tmp -type f -printf "f %s %p\n" | awk '{sub(/^[^ ]+ +[^ ]/,sprintf("%s %10d",$1,$2))}1'
find .
find . -print
find /tmp/dir1 -exec basename {} \;
find /some/directory -type f -exec file -N -i -- {} + | sed -n 's!: video/[^:]*$!!p'
find . -perm -o+w,+s
find ~ -type f -exec file -i {} + | grep video
find . -maxdepth 1 -type f -name '.*' -exec basename {} \;
find . -type f -exec echo {} \;
find ~/some/directory -name "*rb" -exec basename {} \;
diff  --brief --recursive dir1/ dir2/
find  /path/to/directory/* -maxdepth 0 -type d -exec basename {} \;
find /path/to/directory/* -maxdepth 0 -type d -exec basename -a {} +
find /path/to/directory/* -maxdepth 0 -type d -printf '%f\n'
find /usr/local/svn/repos/ -maxdepth 1 -mindepth 1 -type d -exec echo /usr/local/backup{} \;
find . -name "*.txt" -printf "%T@ %p\n" | sort | tail -1
cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1
sudo find . -xdev -type f | cut -d "/" -f 2 | sort | uniq -c | sort -n
find $DIR -name "*.txt" -exec wc -l {} \;
wc -l file.txt | cut -f1 -d" "
ping google.com | awk '{ sent=NR-1; received+=/^.*(time=.+ ms).*$/; loss=0; } { if (sent>0) loss=100-((received/sent)*100) } { printf "sent:%d received:%d loss:%d%%\n", sent, received, loss }'
find . -type f | wc -l
history | cut -d' ' -f4- | sed 's/^ \(.*$\)/\1/g'
history | sed 's/^[ ]*[0-9]\+[ ]*//'
history|awk '{$1="";print substr($0,2)}'
echo `pwd`/`dirname $0`
find /home/kibab -name '*.png' -exec echo '{}' ';'
find -printf '"%h/%f" '
find . -name "*.rb" -type f | xargs -I {} echo Hello, {} !
mount | sed -n -e "s/\/\/mynetaddr on \([^ ]*\).*$/\1/p"
echo <(true)
find /tmp/dir1 -exec echo {} \;
find . |xargs grep abc | sed 's/abc/xyz/g'
find /path/to/directory/* -maxdepth 0 -type d
ping -c 5 -q $host | grep -oP '\d+(?=% packet loss)'
echo "`pwd -P`"
ping -c4 www.google.com | awk '/---/,0'
ping -c 1 8.8.8.8 |  awk 'FNR == 2 { print $(NF-1) }' | cut -d'=' -f2
ping -c 1 8.8.8.8 |  awk 'FNR == 2 { print $(NF-1) }'
echo "$(dirname $(readlink -e $F))/$(basename $F)"
ls -l | head -2 | tail -1
ls -l | tail -n +2 | head -n1
tr -s ' ' | cut -d ' ' -f 2
find $HOME -name '*.ogg' -type f -exec du -h '{}' \;
find . -iname "*.txt" -exec du -b {} +
find . -iname '*.jpg' -type f -printf +%b
find ~/code -name '.*' -prune -o -type f -a -perm /u+x -printf ':%h\n' | sort | uniq | tr -d '\n'
tar tf nginx-1.0.0.tar.gz | xargs dirname | sort | uniq
finger | cut -d ' ' -f1 | sort -u
finger | cut -d ' ' -f1 | sort -u | grep -iv login
finger | tail -n +2 | awk '{ print $1 }' | sort | uniq
who |grep -i admin |cut -c10-20
grep -rl "needle text" my_folder | tr '\n' '\0' | xargs -r -0 file | grep -e ':[^:]*text[^:]*$' | grep -v -e 'executable'
who -b
ping 8.8.8.8 | awk -F"[= ]" '{if($10>50) {cmd="date"; cmd | getline dt; close(cmd) ; print $10, dt}}'
ping 8.8.8.8 | awk '{split($7,a,"[=.]");if (a[2]>58) print a[2], strftime()}'
history | awk '{ print $2 }' | sort | uniq -c |sort -rn | head
history | awk '{print $2}' | awk 'BEGIN {FS="|"}{print $1}' | sort | uniq -c | sort -nr | head
cat files.txt | xargs du -c | tail -1 | awk '{print $1}'
echo $(cat /proc/$$/cmdline)
nl -n ln | sort -u -k 2| sort -k 1n | cut -f 2-
echo `whoami`
whoami
who -m | awk '{print $1}'
seq -f 'some line %g' 500 | nl | sort -R | cut -f2- | head -3
date +"%T"
readlink `pwd`
df -T $dir | tail -1 | awk '{print $2;}'
comm -23 <(sort file1) <(sort file2)|grep -f - file1
sort file_a file_b|uniq -u
comm -3 <(sort set1) <(sort set2) | sed 's/\t//g'
comm -23 file1 file2
comm -2 -3 A B | comm -2 -3 - C | comm -2 -3 - D
comm -23 a.txt b.txt
comm -3 a b
comm -23 second-file-sorted.txt first-file-sorted.txt
comm -23 "File 1" "File 2"
comm -2 -3 <(sort A.txt) <(sort B.txt)
comm -23 <(sort -u A.txt) <(sort B.txt)
comm -3 a b | sed 's/^\t//'
comm -23 a b
comm -13 a b
comm -2 -3 f1 f2
cut -d' ' -f1 file2 | comm -13 - file1
comm -13 first.txt second.txt
who -su | sort | uniq | column
yes "$OPTARG" | head -$opt
echo $modules | column -t | fold | column -t
echo "$opt" | tr -d '"'
date '+%Y' --date='222 days ago'
diff -q "$file" "${file/${dir1}/${dir2}}"
diff -q <(sort set1) <(sort set2)
diff -q <(sort set1 | uniq) <(sort set2 | uniq)
diff --brief -r dir1/ dir2/
diff -arq folder1 folder2
diff --brief -Nr dir1/ dir2/
diff -qr /tmp/dir1/ /tmp/dir2/
diff PATH1/ PATH2/ -rq -X file1
diff -qr dir1 dir2
diff -qr dir1/ dir2/
diff -rq dir1 dir2
diff -qr dir_one dir_two | sort
diff -rqyl folder1 folder2 --exclude=node_modules
pwd | tr '/' '\n'
date -j -v-1d
date +%Y:%m:%d -d "1 day ago"
date +%Y:%m:%d -d "yesterday"
date -d "-1 days" +"%a %d/%m/%Y"
find your/dir -prune -empty -type d
find "your/dir" -prune -empty
find your/dir -prune -empty
echo "$NEWFILE" | sudo tee /etc/apt/sources.list
find "$d" -type f -printf "%T@ :$f %p\n" | sort -nr | cut -d: -f2- | head -n"$m"
du -ksh * | sort -n -r
du -ks * | sort -n -r
ps -aux | grep ^username | awk '{print $2}' | xargs pstree
sort --random-sort $FILE | head -n 1
grep -m1 -ao '[0-9]' /dev/urandom | sed s/0/10/ | head -n1
seq 2000 65000 | sort -R | head -n 1
dig google.com ns | awk 'p{print $5}/^;; ANSWER SECTION:$/{p=1}/^$/{p=0}'
cal -h | cut -c19-20
cal -h | cut -c 4-17
pstree -p 20238 | sed 's/(/\n(/g' | grep '(' | sed 's/(\(.*\)).*/\1/' | tr "\n" ,
pstree -p 20238 | sed 's/(/\n(/g' | grep '(' | sed 's/(\(.*\)).*/\1/'
cal | sed -e 's/^.\{3\}//' -e 's/^\(.\{15\}\).\{3\}$/\1/'
cal -h|sed -r "s/\b$(date|cut -d' ' -f3)\b/*/"
cal
cal $month $year | awk -v day=$day -f cal.awk
cal April 2012 | tee t | more
cal 2 1900
paste <(cal 6 2009) <(cal 6 2010)
pwd | awk -F/ '{print $NF}'
pwd | grep -o "\w*-*$"
pwd | sed 's#.*/##'
cal -h
cal 01 2015 | sed -n '1,2b;/^.\{6\} \{0,1\}\([0-9]\{1,2\}\) .*/ {s//0\1/;s/.*\([0-9]\{2\}\)$/\1/p;q;}'
cal $m $y | sed -e '1,2d' -e 's/^/ /' -e "s/ \([0-9]\)/ $m\/\1/g"
cal | awk 'NR==2 {for (i=1;i<=NF;i++) {sub(/ /,"",$i);a[$i]=i}} NR>2 {if ($a["Tu"]~/[0-9]/) {printf "%02d\n",$a["Tu"];exit}}' FIELDWIDTHS="3 3 3 3 3 3 3 3"
cal | awk 'NR>2 && NF>4 {printf "%02d\n",$(NF-4);exit}'
cal | awk 'NR>2{Sfields=7-NF; if (Sfields == 0 ) {printf "%02d\n",$3;exit}}'
$(dirname $0)
`dirname $0`
echo <(true)
head -c -N file.txt
dirname `find / -name ssh | grep bin | head -1`
echo dirname: $(dirname $mystring)
echo "dirname: `dirname "$0"`"
echo "dirname: `dirname $0`"
ls $PWD/cat.wav
ls -1 | awk  -vpath=$PWD/ '{print path$1}'
ls -d $PWD/*
ls -d -1 $PWD/**
ls -d -1 $PWD/*.*
pstree -p
groups                                        //take a look at the groups and see
groups el                                     //see that el is part of www-data
cp --help
pstree
echo "$(ifconfig)"
w -h $euids
find . -type f -printf '%TY-%Tm-%Td %TH:%TM: %Tz %p\n'| sort -n | tail -n1
find . -type f | sed 's/.*/"&"/' | xargs ls -E | awk '{ print $6," ",$7 }' | sort | tail -1
find . -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" "
ls -1t | head -1
find . -type f -print | xargs -L1 wc -l
find . -type f -print0 | xargs -0L1 wc -l
wc -l *.c
find . -name '*.php' -print0 | xargs -0 wc -l
find . -name '.git' | xargs -n 1 dirname
echo "groups: [ $(groups myuser | sed -e 's/.\+\s\+:\s\+\(.\+\)/\1/g' -e 's/\(\s\+\)/, /g') ]"
ls -a | tee output.file
w | sort
ls -l -- "$dir/$file"
ls -l ${0}
find . -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" " | sed 's/.*/"&"/' | xargs ls -l
echo 'ls -hal /root/ > /root/test.out' | sudo bash
ls -hal /root/ | sudo tee /root/test.out
ls -lb
$ ls -Fltr "./my dir" "./anotherdir"
ls -ld /tmp /tnt | sed 's/^.*$/<-- & --->/'
ls -ld /tmp /tnt
$ ls -Fltr $var
ls -al file.ext
$ ls -Fltr *
$ ls -Fltr
ps -e -orss=,args= | sort -nr | head
ls |& tee files.txt
ls -lR / | tee -a output.file
ls -lR / | tee output.file
echo "The script you are running has basename `basename $0`, dirname `dirname $0`"
ls -l /proc/$$/exe | sed 's%.*/%%'
mktemp -u
wc *.py
cal $(date +"%m %Y") | awk 'NF {DAYS = $NF}; END {print DAYS}'
echo "Number files in SEARCH PATH with EXTENSION:" $(ls -1 "${SEARCHPATH}"/*."${EXTENSION}" | wc -l)
cat $file | cut -c 1-10
echo "$COMMANDS"$'\n'"$ALIASES" | sort -u
groups $(who | cut -d' ' -f 1)
echo this dir: `dirname $BASH_SOURCE`
dirname "/path/to/vm.vmwarevm/vm.vmx"
echo /path/to/vm.vmwarevm/vm.vmx | xargs dirname
$(dirname $(readlink -e ../../../../etc/passwd))
$(dirname $(readlink -e ../../../../etc/passwd))
dirname `readlink -e relative/path/to/file`
ps -aux |  awk '/^username/{print $2}' | xargs pstree
pstree -A -s $$
pstree -sp $$
pstree --show-parents -p $$ | head -n 1 | sed 's/\(.*\)+.*/\1/' | less
pstree -s $ID
pstree -aAp $(ps -ejH | awk -v v1=$mypid '$1==v1 {print $3}')
pstree -p $$
pstree -s $$
pstree -p user
pstree -apl
pstree -a "$PID"
pstree | grep MDSImporte | cut -c 1-${WIDTH}
pstree | grep MDSImporte | less -SEX
sort --random-sort $FILE | head -n 1
$(readlink -f $(dirname "$0"))
pstree -a pid
echo `ls` "`cal`"
dig +noall +answer soa yahoo.com | awk '{sub(/.$/,"",$5);print $5}'
shopt -p globstar
du -h --max-depth=1 . | sort -n -r
w -h  | cut -d' ' -f1 | sort | uniq
echo "0 1 * * * /root/test.sh" | tee -a /var/spool/cron/root
ifconfig | grep HW
grep -r "texthere" .
find . -type f -printf '%T@ %p\n' | sort -n -r | head -${numl} |  cut -f2- -d" " | sed -e 's,^\./,,' | xargs ls -U -l
ls -1 | tail -n +N | head -n1
echo "dirname/readlink: $(dirname $(readlink -f $0))"
cal | awk '$6{date=$6}END{print date}'
echo "$(dirname $(readlink -e $F))/$(basename $F)"
head -c N file.txt
cat filename | awk '{print length, $0}'|sort -nr|head -1
sudo du -hDaxd1 /var | sort -h | tail -n10
history | awk '{ print $2 }' | sort | uniq -c |sort -rn | head
history | awk '{print $2}' | awk 'BEGIN {FS="|"}{print $1}' | sort | uniq -c | sort -nr | head
find . -name '*.php' -type f -exec cat -- {} + | wc -l
find . -name "*.py" -exec wc -l {} \; | awk '{ SUM += $0} END { print SUM }'
find . -type f -exec wc -l {} \; | awk '{ SUM += $0} END { print SUM }'
find . -name '*.c' -print0 |xargs -0 wc -l|grep -v total|awk '{ sum += $1; } END { print "SUM: " sum; }'
cat `find . -name "*.php"` | wc -l
cal | awk 'NR==2 {split($0,a)} {for (i=1;i<=NF;i++) if ($i==date) print a[i]}' FIELDWIDTHS="3 3 3 3 3 3 3 3" date=31
cal | awk -v date=31 'NR == 2 { split($0, header) } NR > 2 { for (i = 1; i <= NF; ++i) if ($i == date) { print header[NR == 3 ? i + 7 - NF : i]; exit } }'
cal | awk -v date=31 -v FIELDWIDTHS="3 3 3 3 3 3 3 3" 'NR==2 {split($0,a)} {for (i=1;i<=NF;i++) if ($i==date) print a[i]}'
date --date="222 days ago" +"%d"
date --date="222 days ago" +"%Y"
date -d "yesterday 13:00" '+%Y-%m-%d'
date --date yesterday "+%a %d/%m/%Y"
find -type f -maxdepth 1 -name 'file*' -print0 | sort -z | xargs -0 bash script.sh
grep -v "#" -R  /var/spool/cron/tabs
read REPLY\?"This is the question I want to ask?"
crontab
pushd `pwd`
pushd $(dirname `which $@`)
pushd $PWD
pushd .
pushd "${line/#\~/$HOME}";
pushd $(dirname $0)
MY_DIR=$(dirname $(readlink -f $0))
bg %1 [puts the job in the background]
dig -t SRV _kerberos._udp.foo.com
bind -q complete
set -u
set -eu
head -c 10 "$0" | tr '\000-\377' '#'
read -rep $'Please Enter a Message:\n' message
read
read -a arr
read -a first
read XPID XUSERID XPRIORITY XVIRTUAL XRESIDENT XSHARED XSTATE XCPU XMEM XTIME XCOMMAND
read VARNAME
read -e -p "Do that? [Y,n]" -i Y input
read -e -p "${myprompt@P}"
read -e -p '> ' $1
read -p "$1 " yn
read -p "Do you wish to install this program?" yn
read -s ENTERED_PASSWORD
read PASSWORD
read -p "$*"
read -p "$1 ([y]es or [N]o): "
read -p "> $line (Press Enter to continue)"
read -p "Press [Enter] key to release lock..."
read -p "Password: " -s SSHPASS
read -r -p "$(echo $@) ? [y/N] " YESNO
read -r a
read -r ans
read -p 'BGG enter something:' -r data
read dir
read -s foobar
read -p " Again? Y/n " i
read -p "$(echo -e 'Please Enter a Message: \n\b')" message
read -p "`echo -e '\nPlease Enter\na Message: '`" message
read -p "Please Enter a Message: `echo $'\n> '`" message
read -p "`echo -e 'Please Enter a Message: \n\b'`" message
read -p "Please Enter a Message: $cr" message
read -s password
read -s -p "Password: " password
read -p "Are you sure you want to continue? <y/N> " prompt
read -r -p "${1:-Are you sure? [y/N]} " response
read -r -p "Are you sure? [y/N] " response
read -r -p "About to delete all items from history that match \"$param\". Are you sure? [y/N] " response
read -p " Enter Here : " text
read -t 0.1 -p "This will be sent to stderr"
read -t 10
read -p "<Your Friendly Message here> : y/n/cancel" CONDITION;
read -p "Are you alright? (y/n) " RESP
read -p "Are you sure you wish to continue?"
read -r -p "Are you sure? [Y/n]" response
read -p "Continue (y/n)?" CONT
read -p "Continue (y/n)?" choice
read -p $'Enter your age:\n'
read -p "Enter your choice: " choice
read -e -i "yes" -p "Enter your choice: " choice
read -p "Is this a good question (y/n)? " answer
read -e
read -p "`pwd -P`\$ " _command
read -p "command : " input_cmd
read -e -p "Enter the path to the file: " -i "/usr/local/etc/" FILEPATH
read -e -p "Enter your choice: " choice
read -e -p "My prompt: " varname
read -rn1
read -r -n 1 -p "${1:-Continue?} [y/n]: " REPLY
read -n1 ans
read -n1 -p "Do that? [y,n]" doit
read -rp $'Are you sure (Y/n) : ' -ei $'Y' key
read -n1 -r -p "Press any key to continue..." key
read -t5 -n1 -r -p 'Press any key in the next five seconds...' key
read -n1 -p "Pick a letter to run a command [A, B, or C for more info] " runCommand
read -d'' -s -n1
read -p "Are you sure? " -n 1 -r
read -p "Are you sure? (y/n) " -n 1
read -t 3 -n 1 -p "Is this a good question (y/n)? " answer
read -n 1 -p "Is this a good question (y/n)? " answer
read line
source "$( dirname "$( which "$0" )" )/lib/B"
awk 'FNR==NR { array[$1]++; next } { n = asorti(array,sort); for (i=1; i<=n; i++) if (sort[i] <= $1 + 10 && sort[i] >= $1 - 10 && $1 != sort[i]) line = (line ? line "," : line) sort[i]; print $0, line; line = "" }' file.txt{,} | column -t
od -vtx1 /dev/midi1
read -u 4 line
dig TXT -f 1.txt
read -n 1 -r
read -n1
read -n 1 c
tr -cs '[:space:]'
read -r -d $'\0' f2
read -d '' line
read -r -d $'\0'
read -n10 -e VAR
du -s $i | read k
history -r "$HISTFILE"     #Alternative: exec bash
history -r
cat /dev/input/mice | od -t x1 -w3
od -A n -N 2 -t u2 /dev/urandom
date --date yesterday "+%a %d/%m/%Y" | read dt
bzip2 -dc input1.vcf.bz2 input2.vcf.bz2 | awk 'FNR==NR { array[$1,$2]=$8; next } ($1,$2) in array { print $0 ";" array[$1,$2] }'
cat
cat -n
cat f.html | grep -o \
inarray=$(echo ${haystack[@]} | grep -o "needle" | wc -w)
find . -type f -exec mv '{}' '{}'.jpg \;
find /path -type f -not -name "*.*" -exec mv "{}" "{}".jpg \;
chmod -R a+rX *
chmod -R +xr directory
chmod -Rf u+w /path/to/git/repo/objects
rm -fR "${TMP}/";
rsync -nvraL test/a/ test/dest --exclude=/b/c/d
sudo chown -R $(whoami):admin /usr/local
sudo chmod -R 777 theDirectory/
sudo chown $(whoami):$(whoami) /usr/local/rvm/gems/ruby-2.0.0-p481/ -R
chown -R $JBOSS_AS_USER:$JBOSS_AS_USER $JBOSS_AS_DIR
chown -R $JBOSS_AS_USER:$JBOSS_AS_USER $JBOSS_AS_DIR/
chown -R tomcat7:tomcat7 webapps temp logs work conf
chown -R user_name folder
chown -R $1:httpd *
chown amzadm.root -R /usr/lib/python2.6/site-packages/
chown amzadm.root -R /usr/lib/python2.6/site-packages/awscli/
chown -R tomcat6 webapps temp logs work conf
chown -R www-data /var/www/.gnome2 /var/www/.config /var/www/.config/inkscape
sudo chown -R $(whoami) /usr/lib/node_modules/
sudo chown -R `whoami` /usr/local
sudo chown -R `whoami` /usr/local/lib
sudo chown -R `whoami` /usr/local/lib/node_modules
sudo chown -R $(whoami) ~/.npm
sudo chown -R `whoami` ~/.npm
chown -R :daemon /tmp/php_session
chown -R :lighttpd /var/lib/php/session
sudo chown -R :laravel ./bootstrap/cache
sudo chown -R :laravel ./storage
chown -R your_user_name.your_user_name 775 /home/el/svnworkspace
chown -R antoniod:antoniod /opt/antoniod/
chown -R antoniod:antoniod /var/antoniod-data/
chown -R your_user_name.your_user_name 775 /workspace
chown user1:user1 -R subdir1
chown user2:user2 -R subdir2
chown user3:user3 -R subdir3
chown "dev_user"."dev_user" -R ~/.ssh/
chown nginx:nginx /your/directory/to/fuel/ -R
chown -R owner:owner public_html
chown -R andrewr:andrewr *
find . -maxdepth 1 -not -name "." -print0 | xargs --null chown -R apache:apache
ls | xargs chown -R apache:apache
sudo chown -R www-data:www-data /var/www
find /mydir -type f -name "*.txt" -execdir chown root {} ';'
sudo chown -R test /home/test
sudo chown -R $USER /usr/local/lib/node_modules
chown ftpuser testproject/ -R
chown -R nobody upload_directory
sudo chown -R $USER ~/tmp
sudo chown -R  $USER:$GROUP /var/lib/cassandra
sudo chown -R  $USER:$GROUP /var/log/cassandra
chown -R ${JBOSS_USER}: $JBOSS_LOG_DIR
sudo chown -R ec2-user:apache /vol/html
chown -R user:www-data yourprojectfoldername
ls -d * | grep -v foo | xargs -d "\n" chown -R Camsoft
sudo chown -R xxx /Users/xxx/Library/Developer/Xcode/Templates
chown -R root:root /var/cache/jenkins
chown -R root:root /var/lib/jenkins
chown -R root:root /var/log/jenkins
chgrp -R www-data /var/tmp/jinfo
find . -type d | sed -e 's/^\.\///g' -e 's/^\./avoid/g' | grep -v avoid | awk '{print $1"\t"$1}' | xargs chgrp
find . -type d | sed -e 's/^\.\///g' | awk '{print $1, $1}' | xargs chgrp
chgrp -R fancyhomepage /home/secondacc/public_html/community/
chgrp -R apache_user files
chgrp -R my_group files
chgrp -R project_dev /home/user1/project/dev
chgrp -R git .git
chgrp -R shared_group /git/our_repos
chgrp -R GROUP /path/to/repo
chgrp -R repogroup .
find . -group X_GNAME -exec chgrp Y_GNAME {} +
chgrp -R admin *
chgrp -R git ./
chgrp -R $GROUP $PATH_TO_OUTPUT_FOLDER
find ${WP_ROOT}/wp-content -exec chgrp ${WS_GROUP} {} \;
chgrp --recursive website public_html
gzip -kr .
find $2 -type f -exec bzip2 {} \;
cp -R "$1" "$2"
cp -Rp "$appname.app" Payload/
cp -r $1 $2
cp -r ../include/gtest ~/usr/gtest/include/
cp -R SRCFOLDER DESTFOLDER/
sudo cp -a include/gtest /usr/include
cp -nr src_dir dest_dir
cp -rs /mnt/usr/lib /usr/
cp -rv `ls -A | grep -vE "dirToExclude|targetDir"` targetDir
cp -r `ls -A | grep -v "c"` $HOME/
cp -Rvn /source/path/* /destination/path/
yes | cp -rf /zzz/zzz/* /xxx/xxx
cp * .hiddendir -R
rsync -rvv /path/to/data/myappdata/*.txt user@host:/remote/path/to/data/myappdata/
rsync -u -r --delete dir_a dir_b
rsync -u -r --delete dir_b dir_a
rsync --recursive emptydir/ destination/newdir
rsync -rcn --out-format="%n" old/ new/
rsync -Prt --size-only original_dir copy_dir
rsync -rvc --delete --size-only --dry-run source dir target dir
scp -r prod:/path/foo /home/user/Desktop
scp -r user@your.server.example.com:/path/to/foo /home/user/Desktop/
rsync -rvv *.txt user@remote.machine:/tmp/newdir/
rsync --recursive --exclude 'foo' * "$other"
find /path/to/source -type d | cpio -pd /path/to/dest/
find demo -depth -name .git -prune -o -print0 | cpio -0pdv --quiet demo_bkp
scp -Bpqr /tdggendska10/vig-preview-dmz-prod/docs/sbo/pdf/*ela*L1*TE* dalvarado@localhost:/var/www/html/sbo/2010/teacher/ela/level1
rsync -r --verbose --exclude 'exclude_pattern' ./* /to/where/
rsync -zvr --exclude="*" --include="*.sh" --include="*/" "$from" root@$host:/home/tmp/
rsync --recursive --prune-empty-dirs --include="*.txt" --filter="-! */" dir_1 copy_of_dir_1
find . -type f -exec scp {} hostname:/tmp/{} \;
scp -r A D anotherhost:/path/to/target/directory
rsync -rvv /path/to/data/myappdata user@host:/remote/path/to/data/myappdata
rsync -rvv --recursive /path/to/data/myappdata user@host:/remote/path/to/data/newdirname
scp -r myServer:/something
cp -Rvn /source/path/* /destination/path/
scp -i /path/to/your/.pemkey -r /copy/from/path user@server:/copy/to/path
find . -path './pma' -prune -o -path './blog' -prune -o -path './punbb' -prune -o -path './js/3rdparty' -prune -o -print | egrep '\.php|\.as|\.sql|\.css|\.js' | grep -v '\.svn' | xargs cat | sed '/^\s*$/d' | wc -l
find . -type f -name '*.c' -exec cat {} \; | sed '/^\s*#/d;/^\s*$/d;/^\s*\/\//d' | wc -l
find . -name .svn -delete
find  . -name \*.txt -print -exec cat {} \;
find posns -type f -exec split -l 10000 {} \;
find . -type f  -mtime +7 | tee compressedP.list | xargs -I{} -P10 compress {} &
cat searches.txt| xargs -I {} -d, -n 1 grep  -r {}
find . -name "*zip" -type f | xargs ls -ltr | tail -1
find . -type f -print0|xargs -0 ls -drt|tail -n 1
find . -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" "
find . -type f | xargs ls -ltr | tail -n 1
find . -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" " | sed 's/.*/"&"/' | xargs ls -l
find . -type f -printf '%TY-%Tm-%Td %TH:%TM: %Tz %p\n'| sort -n | tail -n1
grep -rnw "pattern"
find . -type f \( -name "*.pas" -o -name "*.dfm" \) -print0 | xargs --null grep --with-filename --line-number --no-messages --color --ignore-case "searchtext"
find ./ -name "file_pattern_name"  -exec grep -r "pattern" {} \;
find . -name "*.pdf" -print0 | xargs -0 rm -rf
find . -name '*.pdf' -exec rm -f {} \;
find . -name "*.pdf" -exec rm {} \;
find . -name "*.pdf" -print0 | xargs -0 rm
find . -name '*.pdf' -exec rm {} +
find . -name "*.png" -mtime +50 -exec rm {} \;
fgrep --include='STATUS.txt' -rl 'OPEN' | xargs -L 1 dirname
find -type f -name "STATUS.txt" -exec grep -q "OPEN" {} \; -exec dirname {} \;
find / -name \*.dbf -print0 | xargs -0 -n1 dirname | sort | uniq
find . -name "*.txt" | xargs grep -i "text_pattern"
find ./ -iname "*.bz2" -exec bzip2 -d {} \;
find ./ -type f -exec grep -H 'text-to-find-here' {} \;
find / -type f | xargs grep 'text-to-find-here'
find . | xargs grep 'word' -sl
grep -r OPEN * | awk '{split($1, path, ":"); print path[1]}' | xargs -I{} dirname {}
find . -type f | egrep -v '\.bz2' | xargs bzip2 -9 &
find / -type f -exec grep -l "text-to-find-here" {} \;
find ~/ -type f -exec grep -H 'text-to-find-here' {} \;
find . -depth -type f -not -name *.itp -and -not -name *ane.gro -and -not -name *.top -exec rm '{}' +
find . -type f  -mtime +7 | tee compressedP.list | xargs -I{} -P10 compress {} &
grep -inr "Text" folder/to/be/searched/
grep -rnw `pwd` -e "pattern"
find . -name '.git' | xargs dirname
find . -type f -print0 | xargs -0 -n1 -P4 bzip2
find . -type f -exec bzip2 {} +
find /path/to/dir -type f -exec bzip2 {} \;
find . -type f -name some_file_name.xml -exec grep -H PUT_YOUR_STRING_HERE {} \;
find . -name '*.js' | grep -v excludeddir
find -iname example.com | grep -v beta
find .  \( ! -path "./output/*" \) -a \( -type f \) -a \( ! -name '*.o' \) -a \( ! -name '*.swp' \) | xargs grep -n soc_attach
grep "class foo" **/*.c
find ./ -type f | xargs grep "foo"
grep -r --include "*.txt" texthere .
egrep -R "word-1|word-2” directory-path
egrep -w -R "word-1|word-2” directory-path
grep -r -H "text string to search” directory-path
grep [option] "text string to search” directory-path
grep -insr "pattern" *
grep --include="*.xxx" -nRHI "my Text to grep" *
tree -F coreutils-8.9 | sed -r 's|── (.*)/$|── DIR: \1|'
mount -v | grep smbfs | awk '{print $3}' | xargs ls -lsR
diff -rq /dir1 /dir2 | grep -E "^Only in /dir1.*" | sed -n 's/://p' | awk '{print $3"/"$4}'
tree
ls **/*.py **/*.html
ls -ldt $(find .)
ls -ld $(find .)
find /path/to/srcdir -type f -print0 | xargs -0 -i% mv % dest/
tree -d
tree .
tree -a .
tree $absolute/path/of/your/dir
find $(pwd) -name \*.txt -print
find ./ -type f -print -exec grep -n -i "stringYouWannaFind" {} \;
find . -name "*.class" -print0 | xargs -0 -n1 dirname | sort --unique
grep -RIl "" .
yes n | rm -r *.txt
yes | rm -r *.txt
rm -r $TMPDIR
rm -r classes
find X -depth -type d -exec rmdir {} \;
find -depth -type d -empty -exec rmdir {} \;
find . -depth -type d -empty -exec rmdir {} \;
find -type d -empty -exec rmdir -vp --ignore-fail-on-non-empty {} +
find . -name "FILE-TO-FIND" -exec rm -rf {} +
find . -name "FILE-TO-FIND" -exec rm -rf {} \;
find . -depth -name .svn -exec rm -fr {} \;
find . -name .svn -exec rm -rf {} +
find . -name .svn -exec rm -rf {} \;
find . -name .svn | xargs rm -fr
find . -name .svn |xargs rm -rf
rm -rf /usr/local/{lib/node{,/.npm,_modules},bin,share/man}/npm*
find [path] -type f -not -name 'EXPR' | xargs rm
find . -type f -not -name '*txt' | xargs rm
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
find . -name "*.pyc" | xargs -I {} rm -v "{}"
find . -name "*.pyc" -exec rm -rf {} \;
find . -name "*.pyc"|xargs rm -rf
find . -name '*.pyc' -print0 | xargs -0 rm
rm **/*.pyc
rm `find . -name \*.pyc`
find . -type f -name "*.py[c|o]" -exec rm -f {} +
find ./ -type f -name '*.r*' -delete -printf "%h\0" | xargs -0 rmdir
find . -name \*.xyz -exec rm {} \;
rm `find ./ -name '.DS_Store'` -rf
find . -name "._*" -print0 | xargs -0 rm -rf
find a -type f -name '4' -delete -printf "%h\0" | xargs -0 -r rmdir
rm /var/www/html/**/_* /var/www/html/**/.DS_Store
find . -iname '*.pyc' -print0 | xargs -0 --no-run-if-empty  rm
find . -name .svn -exec rm -v {} \;
find /home/ubuntu/wars -type f -name "*.war" -exec rm {} \\;
find . -type d -name .svn -print0|xargs -0 rm -rf
find /your/target/path/ -type f -exec rename 's/special/regular/' '{}' \;
find . -name "*.t1" -exec rename 's/\.t1$/.t2/' '{}' \;
grep -r "string here" * | tee >(wc -l)
find . -type d -iname '*foo*' -depth -exec rename 's@Foo@Bar@gi' {} +
find . -type f \! -name '*.xml' -print0 | xargs -0 rename 's/$/.xml/'
find . -name "*_test.rb" | xargs -s 1000000 rename s/_test/_spec/
find . -name "*_test.rb" | xargs -P 4 rename s/_test/_spec/
find . -name "*_test.rb" | xargs rename s/_test/_spec/
find ./dir1 -type f -exec basename {} \;
find . -exec file {} \;
grep -Ri "pattern" * | awk '{if($1 != "Binary") print $0}'
find ~/test -type d -exec basename {} \;
chmod -R 777 ../tools
chmod -R 755 /directory
chmod 755 /folder -R
sudo chmod 755 -R /opt/lampp/htdocs
sudo chmod 755 -R /whatever/your/directory/is
zcat -r /some/dir/here | grep "blah"
date -j -f "%a %b %d %H:%M:%S %Y %z" "Sat Aug 09 13:37:14 2014 +1100"
mount / -o remount,rw
mount -n -o remount /
mount -o remount,rw -t yaffs2 /dev/block/mtdblock3 /system
mount -o remount,ro -t yaffs2 /dev/block/mtdblock3 /system
sudo mount -o remount -o exec /dev/sda7
mount -o remount,size=40G /dev/shm
mount -o rw,remount /dev/stl12 /system
mount -o ro,remount /dev/stl12 /system
sudo mount -i -o remount,suid /home/evgeny
mount /media/Working/ -oremount,umask=000
mount /mnt/mountpoint -oremount,ro
mount /mnt/mountpoint -oremount,rw
mount -o remount,ro /path/to/chroot/jail/usr/bin
mount -o remount,ro /system
mount -o remount,rw /system
mount -o rw,remount /system
mount -o remount,ro /system
mount -n -o remount -t extX /dev/hdaX /
mount -o rw,remount -t rfs /dev/stl12 /system
mount -o rw,remount -t yaffs2 /dev/block/mtdblk4 /system
mount -o ro,remount -t yaffs2 /dev/block/mtdblk4 /system
mount -oremount /
mount /outside /inside -o bind
mount -o rw,remount -t rootfs /
comment=$(cat /proc/sys/kernel/random/uuid | sed 's/\-//g')
echo -e "test1\ntest2\ntest3" |tr -d '\n' |grep "test1.*test3"
cat infile | sed 's/\r$//' | od -c
rename _dbg.txt .txt **/*dbg*
bind -r '\e'
find . -name ".DS_Store" -exec rm {} \;
find `pwd` -name core -ctime +4 -execdir /bin/rm -f {} \;
machnum=$(hostname | sed 's/^machine//')
rev input | uniq -f1 | rev
find . -iname CVS -type d | xargs rm -rf
find . -name core -ctime +4 -exec /bin/rm -f {} \;
find /tmp -name core -type f -print | xargs /bin/rm -f
find /usr/ports/ -name work -type d -print -exec rm -rf {} \;
find . \( -name a.out -o -name '*.o' -o -name 'core' \) -exec rm {} \;
find . \( -name '*.bak' -o -name *.backup \) -type f -atime +30 -exec rm '{}' ';'
find . -type f -name \*.bak -print0 | xargs -0 rm -v
find . -name '*.doc' -exec rm "{}" \;
find -name '*.log' -delete
find ./ -name '*.log' -print0 | xargs -0 rm
find ./ -name '*.log' | xargs rm
find tmp -maxdepth 1 -name '*.mp3' -maxdepth 1 | xargs    -n1 rm
find tmp -maxdepth 1 -name '*.mp3' -maxdepth 1 | xargs    rm
find tmp -maxdepth 1 -name *.mp3 -print0 | xargs    -0 rm
rm `find tmp -maxdepth 1 -name '*.mp3'`
find $backup_path/*.sql -mtime +5 -exec rm -f {} \;
find $backup_path/* -name *.sql -mtime +30 -exec rm {} \;
find . -name "*.swp"|xargs rm
find . -name "*.swp" -print0|xargs -0 rm
find . -name "*.swp"-exec rm -rf {} \;
find /tmp -name "*.tmp" -print0 | xargs -0 rm
find /tmp -name "*.tmp" | xargs rm
find $HOME/. -name *.txt -ok rm {} \;
find /home/u20806/public_html -maxdepth 1 -mmin +5 -type f -name "*.txt" -delete
find /home/u20806/public_html -maxdepth 1 -mmin +5 -type f -name "*.txt" ! -name "robots.txt" -delete
find /home/u20806/public_html -name "robots.txt" -o -maxdepth 1 -mmin +5 -type f -name "*.txt" -delete
find . -name '*bak' -exec rm -i {} \;
find . -name '*~' -ok rm {} \;
find . -name '*.gz' -type f -printf '"%p"\n' | xargs rm -f
find /home/luser -type f -name '*.mpg' -exec rm -f {} \;
find /home/luser -type f -name '*.mpg' -print0 | xargs -0 rm -f
find /home/luser -type f -name '*.mpg' | tr "\n" "\000" | xargs -0 rm -f
find /home/luser -type f -name '*.mpg' | xargs rm -f
find /var/www/*.php -type f -exec rm {} \;
find . -name "new*.sh" -exec rm -f '{}' \+
find . -name "new*.sh" -exec rm -f '{}' \;
find . -name "t*.sh" -exec rm -vf '{}' \;
find /tmp -name "*.tmp" -print0 | xargs -0 rm
find /tmp -name "*.tmp" | xargs rm
find /full/path/dir -name '*.txt' -exec /bin/rm {} \;
find /full/path/dir -name '*.txt' -print0 | xargs -0 rm
find . -name "*.txt" -delete
find . -name "*.txt" -exec rm {} +
find . -name "*.txt" -exec rm {} \;
find . -name "*.txt" -print0 | xargs -0 rm
find . -name "*.txt" | xargs rm
find -name "*\ *.txt" | xargs rm
find . -name 'CVS' -type d -exec rm -rf {} \;
find . -type d -name CVS -exec rm -r {} \;
find . -name Thumbs.db -exec rm {} \;
find dir -name \\*~ -exec rm {} +
find . \( -name a.out -o -name '*.o' -o -name 'core' \) -exec rm {} \;
find -L /usr/ports/packages -type l -delete
find -L /usr/ports/packages -type l -delete
cat data.csv | rev | cut -d, -f-5 | rev
tr -cd ";0-9"
find ~/ -name 'core*' -exec rm {} \;
find /path/to/dir -name "test" -type d -delete
find /path/to/dir -name "test" -type d -exec rm -rf {} \;
find -name "test" -type d -delete
find -path "*/test" -type d -delete
find -path "*/test/*" -delete
find . -name test -type d -exec rm -r {} +
find . -name test -type d -exec rm -r {} \;
find . -name test -type d -print0|xargs -0 rm -r --
find $LOGDIR -type d -mtime +5 -exec rm -f {} \;
find \! -name . -type d -print0 | xargs -0 rmdir
find ./ -type d -size 0c -print | xargs rmdir
find /tmp -type f -empty -print | xargs rm -f
find ./ -type f -empty -print0 | xargs -0 rm
find ./ -type f -size 0c -print | xargs rm
find . -depth  -type d  -empty -exec rmdir {} \;
find $HOME \( -name a.out -o -name '*.o' \) -atime +7 -exec rm {} \;
find . -name test -delete
find . -name test -exec rm -R "{}" \;
find . -name test -exec rm {} \;
find /home -name Trash -exec rm {} \;
yes | /bin/rm -i *
find /home/foo \( -name '.DS_Store' -or -name '._.DS_Store' -or -name '._*' -or -name '.TemporaryItems' -or -name '.apdisk' \) -exec rm -rf {} \;
/usr/bin/find /home/user/Series/ -iname "*sample*" -exec rm {} \;
find /home/user/Series/ -iname '*sample*' -exec rm {} \;
find -type f -printf %P\\n | sort | comm -3 MANIFEST - | xargs rm
find . -name "* *" -exec rm -f {} \;
find . | egrep -v "\.tex|\.bib" | xargs rm
find . | grep -v "excluded files criteria" | xargs rm
find -iname '*~' | xargs rm
find / -type f -print0 | xargs -0 grep -liwZ GUI | xargs -0 rm -f
find . -name not\* -print0 | xargs -0 rm
find . -name not\* | tr \\n \\0 | xargs -0 rm
find . -name not\* | xargs -d '\n' rm
find $backup_path/* -mtime +30 -exec rm {} \;
find /myfiles -atime +30 -exec rm {} ;
find . -name abc.xxx -exec rm {} \;
find ~/backups/mydatabasename/* -mtime +30 -exec rm {} \;
find . -mtime +10 | xargs rm
find . -name '*[+{;"\\=?~()<>&*|$ ]*' -exec rm -f '{}' \;
find . -name FOLDER1 -prune -o -name filename -delete
find . \( -name junk -o -name dummy \) -exec rm '{}' \;
find / \( -name tmp -o -name '*.xx' \) -atime +7 -exec rm {} \;
find / -newerct '1 minute ago' -print | xargs rm
find /file/path ! -newermt "Jul 01" -type f -print0 | xargs -0 rm
find / -type f -print0 | xargs -0 grep -liwZ GUI | xargs -0 rm -f
find /mnt/zip -name "*prefs copy" -print | xargs rm
find . -type f -mtime +3 –exec rm –f {} \;
find "$DIR" -type f -atime +5 -exec rm {} \;
find /home/user/Maildir/.SPAM/cur -type f -exec rm '{}' +
find /home/user/Maildir/.SPAM/cur -type f -exec rm -f '{}' '+'
find /home/user/Maildir/.SPAM/cur -type f | xargs rm
find /myfiles -atime +30 -exec rm {} \;
find -exec rm '{}' +
find /home/peter -name no-such-thing* |xargs rm
find /home/peter -name *~ -print0 |xargs -0 rm
find /home/peter -name *~ |xargs rm
find . -name 'spam-*' | xargs rm
find ./js/ -type f -name "*.js" | xargs rm -f
find . -type f -name "*.txt" -exec rm {} \; -print
find . -type f -name "*.txt" -print|xargs rm
find . -name "vmware-*.log" -exec rm '{}' \;
find . -name vmware-*.log -delete
find . -name vmware-*.log -print0 | xargs -0 rm
find . -name vmware-*.log | xargs -i rm -rf {}
find . -name vmware-*.log | xargs rm
find . -name "*.c" -print0 | xargs -0 rm -rf
find . -name "*.c" | xargs rm -rf
find -mindepth 1 -depth -print0 | grep -vEzZ '(\.git(/|$)|/\.gitignore$)' | xargs -0 rm -rvf
find . -name libEGL* | xargs rm -f
find . -name libGLE* | xargs rm -f
find -name "*" | xargs rm -f
find /path -type f -exec rm '{}' +
find /path -type f -exec rm '{}' \;
find /path -type f -print | xargs rm
find . -type f ! -regex ".*/\(textfile.txt\|backup.tar.gz\|script.php\|database.sql\|info.txt\)" -delete
find . -type f -mtime 1 -exec rm {} +
find . -type f -newermt "Aug 10" ! -newermt "Aug 17" -exec rm {} \;
find . -type f -not -name '*ignore1' -not -name '*ignore2' | xargs rm
find . -type f -not -name '*ignore1' -o -not -name '*ignore2' | xargs rm
find . -type f -not -name '*txt' -print0 | xargs -0 rm --
find ./ -type f -exec rm -rf {} \;
find ~/Books -type f -name Waldo -exec rm {} \;
find "$DIR" -type f -atime +5 -exec rm {} \;
find /var/log/remote/ -daystart -mtime +14 -type f -exec rm {} \;
find "$DIR" -type f -atime +5 -exec rm {} \;
find /old/WordPress/ -type f -regex ".*\.\(php\|css\|ini\|txt\)" -exec rm {} \;
find /tmp -type f -name '*' -mtime +7 -print0 | xargs -0 rm -f
tr -d ' '
find . -maxdepth 1 -type d \( ! -name "bar" -a ! -name "foo" -a ! -name "a" -a ! -name "b" \) -delete
find $HOME/. -name *.txt -ok rm {} \;
diff -rq /dir1 /dir2 | grep -E "^Only in /dir1.*" | sed -n 's/://p' | awk '{print $3"/"$4}' xargs -I {} rm -r {}
find tmp -maxdepth 1 -name '*.mp3' -maxdepth 1 | xargs -n1 rm
find tmp -maxdepth 1 -name '*.mp3' -maxdepth 1 | xargs rm
find tmp -maxdepth 1 -name *.mp3 -print0 | xargs -0 rm
find . -name "vmware-*.log" -exec rm '{}' \;
find . -name vmware-*.log -delete
find . -name vmware-*.log | xargs rm
find . -name vmware-*.log -print0 | xargs -0 rm
find . -name vmware-*.log | xargs -i rm -rf {}
cat infile.txt | tr -d "[:space:]" | fold -80
find / -type f -print0 | xargs -0 grep -liwZ GUI | xargs -0 rm -f
sed '/^$/d;s/ /\//g' struct.txt | xargs mkdir -p
basename /home/jsmith/base.wiki .wiki
path=$(basename $path)
find /media/1Tb/videos -maxdepth 1 -type d -mtime +7 -exec rm -rf {} \;
awk '{print(NR"\t"$0)}' file_name | sort -t$'\t' -k2,2 | uniq --skip-fields 1 | sort -k1,1 -t$'\t' | cut -f2 -d$'\t'
awk '{print(NR"\t"$0)}' file_name | sort -t$'\t' -k2,2 | uniq -u --skip-fields 1 | sort -k1,1 -t$'\t' | cut -f2 -d$'\t'
nl -w 8 "$infile" | sort -k2 -u | sort -n | cut -f2
variable=$(echo "$variable" | tr ' ' '\n' | nl | sort -u -k2 | sort -n | cut -f2-)
sort | uniq -u | xargs -r rm
find -type d -exec rmdir --ignore-fail-on-non-empty {} + ;
find /srv/${x} -type d -empty -exec rmdir {} \;
find /srv/abc/ -type d -empty -exec rmdir {} \;
find . -depth -empty -type d -delete
rmdir --ignore-fail-on-non-empty newBaseDir/Data/NewDataCopy
rm -ri *
find -mindepth 1 -maxdepth 1 -print0 | xargs -0 rm -rf
rm -rf *
find . -maxdepth 1 | grep -v "exclude these" | xargs rm -r
echo '1/2 [3] (27/03/2012 19:32:54) word word word word 4/5' | sed -e 's/(.*)//' -e 's/[^0-9]/ /g' | column -t
find . -inum 31246 -exec rm [] ';'
find ~/junk  -name 'cart[4-6]' -exec rm {}  \;
find . -name "-F" -exec rm {} \;
find / -nouser -exec rm {} +
find / -nouser -exec rm {} \;
find / -nouser -ok rm {} \;
find ~ -atime +100 -delete
find ~/clang+llvm-3.3/bin/ -type f -exec basename {} \; | xargs rm
find . -type f -mtime +31 -print0 | xargs -0 -r rm -f
find . -name "file?" -exec rm -vf {} \;
find  -name '*-*x*.*' | xargs rm -f
find /path/to/files* -mtime +5 -exec rm {} \;
find /work \( -fprint /dev/stderr \) , \( -name 'core' -exec rm {} \; \)
find . -type f -size +1M -exec rm {} +
find . -size -1M -exec rm {} \;
find . -type f -size -1M -exec rm {} +
find /mnt/zip -name "*prefs copy" -print0 | xargs -0 -p /bin/rm
find . -name "* *" -exec rm -f {} \;
find . -inum $inum -exec rm {} \;
find -regex '^.*/[A-Za-z]+-[0-9]+x[0-9]+\.[A-Za-z]+$' | xargs echo rm -f
find . -type f -name "Foo*" -exec rm {} \;
echo $filename | rev | cut -f 2- -d '.' | rev
find . -name "*.*" -type f -exec grep -l '<img-name>-<width:integer>x<height:integer>.<file-ext> syntax' {} \; | xargs rm -f
ssh-keygen -f "/root/.ssh/known_hosts" -R gitlab.site.org
find /path/to/junk/files -type f -mtime +31 -exec rm -f {} \;
find /path/to/junk/files -type f -mtime +31 -print0 | xargs -0 -r rm -f
echo t1_t2_t3_tn1_tn2.sh | rev | cut -d_ -f3- | rev
echo "   wordA wordB wordC   " | sed -e 's/^[ \t]*//' | sed -e 's/[ \t]*$//'
echo "$string" | sed -e 's/^[ \t]*//' | sed -e 's/[ \t]*$//'
grep -v 'kpt#' data.txt | nl -nln
paste -sd "" file.txt
find . -name '*~' -print0 | xargs -0 rm
find /tmp/ -ctime +15 -type f -exec rm {} \;
find . -type f -print0 | xargs -0 -n1 echo rm | sh -x
find -type f  |  grep -P '\w+-\d+x\d+\.\w+$' | sed -re 's/(\s)/\\\1/g' | xargs rm
find -type f |  grep -P '\w+-\d+x\d+\.\w+$' | xargs rm
find . -iname *.js -type f -exec sed 's/^\xEF\xBB\xBF//' -i.bak {} \; -exec rm {}.bak \;
find . -type f -exec sed '1s/^\xEF\xBB\xBF//' -i.bak {} \; -exec rm {}.bak \;
find sess_* -mtime +2 -exec rm {} \;
echo aa | wc -l | tr -d ' '
DIR_PATH=`readlink -f "${the_stuff_you_test}"`
find -name "123*.txt" -exec rename 's/^123_//' {} ";"
rename 's/^123_//' *.txt
find /home -type f -name "*.ext" -exec sed -i -e "s/\r$//g" {} \;
find /home -type f -name "*.ext" -exec sed -i -e "s/\x0D$//g" {} \;
find /home -type f -name "*.ext" -exec sed -i -e 's/^M$//' {} \;
find . -type f -regex ".+-[0-9]+x[0-9]+\.jpg" -exec rm -rf {} \;
find . -type f -regex ".+-[0-9]+x[0-9]+\.jpg" | xargs rm
find -inum 752010 -exec rm {} \;
find ~/ -atime +100 -exec rm -i {} ;
sudo rm -rf bin/node bin/node-waf include/node lib/node lib/pkgconfig/nodejs.pc share/man/man1/node
rm -r bin/node bin/node-waf include/node lib/node lib/pkgconfig/nodejs.pc share/man/man1/node.1
find /tmp -type f \( -name '*.txt' \) |cut -c14- | nl
history | cut -c 8-
cat $filename | rev | cut -c 3- | rev
echo 987654321 | rev | cut -c 4- | rev
echo "filename.gz"     | sed 's/^/./' | rev | cut -d. -f2- | rev | cut -c2-
sed 's/^/./' | rev | cut -d. -f2- | rev | cut -c2-
echo "mpc-1.0.1.tar.gz" | sed -r 's/\.[[:alnum:]]+\.[[:alnum:]]+$//'
echo $path | rev | cut -d'/' -f4- | rev
sed '/pattern to match/d' ./infile
echo | ssh-keygen -P ''
ssh-keygen -f ~/.ssh/id_rsa -P ""
PATH=$(echo $PATH | tr ":" "\n" | grep -v $1 | tr "\n" ":")
find -maxdepth 1 -type f -newermt "Nov 22" \! -newermt "Nov 23" -delete
find ./ -type f -newer /tmp/date.start ! -newer /tmp/date.end -exec rm {} \;
find -type f -newermt "Nov 21" ! -newermt "Nov 22" -delete
find . -not \( -name .svn -prune -o -name .git -prune \) -type f -exec sed -i 's/[:space:]+$//' \{} \;  -exec sed -i 's/\r\n$/\n/' \{} \;
find dir -not -path '.git' -iname '*.py' -print0 | xargs -0 sed --in-place=.bak 's/[[:space:]]*$//'.
find . -name '*.rb' | xargs -I{} sed -i '' 's/[[:space:]]*$//g' {}
find . \( -name *.rb -or -name *.html -or -name *.js -or -name *.coffee -or -name *.css -or -name *.scss -or -name *.erb -or -name *.yml -or -name *.ru \) -print0 | xargs -0 sed -i '' -E "s/[[:space:]]*$//"
find . -not \( -name .svn -prune -o -name .git -prune \) -type f -exec sed -i "s/[[:space:]]*$//g" "{}" \;
find . -not \( -name .svn -prune -o -name .git -prune \) -type f -print0 | xargs -0 sed -i '' -E "s/[[:space:]]*$//"
find . -not \( -name .svn -prune -o -name .git -prune \) -type f -print0 | xargs -0 file -In | grep -v binary | cut -d ":" -f1 | xargs -0 sed -i '' -E "s/[[:space:]]*$//"
find . -type f -not -iwholename '*.git*' -print0  | xargs -0 sed -i .bak -E "s/[[:space:]]*$//"
find dir -type f -exec sed -i 's/ *$//' '{}' ';'
find dir -type f -print0 | xargs -0 sed -i .bak -E "s/[[:space:]]*$//"
find dir -type f -print0 | xargs -r0 sed -i 's/ *$//'
find . -type f -name '*' -exec sed --in-place 's/[[:space:]]\+$//' {} \+
find . -type f -name '*.txt' -exec sed --in-place 's/[[:space:]]\+$//' {} \+
find . -iname '*.txt' -type f -exec sed -i '' 's/[[:space:]]\{1,\}$//' {} \+
sed -r 's/((:[^: \t]*){3}):[^ \t]*/\1/g' file | column -t
find /mydir -atime +100 -ok rm {} \;
find /mydir -atime +100 -ok rm {} \;
sudo rm /var/lib/mongodb/mongod.lock
rm -rf folderName
rm foo
rmdir latest
ls -t *.log | tail -$tailCount | xargs rm -f
nl -nrz -w10 -s\; input | sed -r 's/55//; s/([0-9]{2})-([0-9]{2})-([0-9]{4})/\3\2\1/'
nl -nrz -w10 -s\; input | sed -E 's/55//; s/([0-9]{2})-([0-9]{2})-([0-9]{4})/\3\2\1/'
find /Users -type d -iname '*.bak' -print0 | xargs -0 rmdir
find /path/to/the/folder -depth -type d -print0 | xargs -0 rmdir
find . -type d -empty -exec rmdir "{}" \;
find . -type d -exec rmdir {}\;
find . -newer first -not -newer last -type d -print0 |  xargs -0 rmdir
find /foo/bar -type d -depth -exec rmdir -p {} +
find /thepath -type d -empty -print0 | xargs -0 rmdir -v
find $homeDirData -type d -mmin +10 -print0 | xargs -0 rmdir
find "$DELETEDIR" -mindepth 1 -depth -type d -empty -exec rmdir "{}" \;
ls -tp | grep -v '/' | tail -n +"$1" | xargs -I {} rm -- {}
ls -tp | grep -v '/$' | tail -n +6 | tr '\n' '\0' | xargs -0 rm --
ls -tp | grep -v '/$' | tail -n +6 | xargs -I {} rm -- {}
ls -tQ | tail -n+4 | xargs rm
ls -tp | grep -v '/$' | tail -n +6 | xargs -d '\n' rm --
find . -maxdepth 1 -type f | xargs -x ls -t | awk 'NR>5' | xargs -L1 rm
find . -maxdepth 1 -type f -printf '%T@ %p\0' | sort -r -z -n | awk 'BEGIN { RS="\0"; ORS="\0"; FS="" } NR > 5 { sub("^[0-9]*(.[0-9]*)? ", ""); print }' | xargs -0 rm -f
ls -C1 -t| awk 'NR>5'|xargs rm
ls -tr | head -n -5 | xargs rm
rm `ls -t | awk 'NR>5'`
rm -v *.bak
rm -f A*.pdf
sudo rm -rf /usr/local/bin/npm /usr/local/share/man/man1/node* /usr/local/lib/dtrace/node.d ~/.npm ~/.node-gyp /opt/local/bin/node opt/local/include/node /opt/local/lib/node_modules
rm -f *.pdf
ls | xargs rmdir
rmdir ed*
rmdir edi edw
rm -d symlink
rmdir --ignore-fail-on-non-empty $newBaseDir/Data/NewDataCopy
ls -1|grep -v -e ddl -e docs| xargs rm -rf
rm junk1 junk2 junk3
rm -f ~/.android/adbkey ~/.android/adbkey.pub
finalName=$(basename -- "$(dirname -- "$path")")
finalName=$(dirname ${path#*/})
echo 'test/90_2a5/Windows' | xargs dirname | xargs basename
find . -type d | xargs rmdir
find . -name ".DS_Store" -print0 | xargs -0 rm -rf
find . -iname "Thumbs.db" -print0 | xargs -0 rm -rf
head -n -2 myfile.txt
tac file | sed -e '/./,$!d' | tac | sed -e '/./,$!d'
sudo mv /usr/bin/php /usr/bin/~php
mv Tux.png .Tux.png
mv blah1 blah1-new
mv blah2 blah2-new
mv fghfilea jklfilea
rename -v 's#/file##' v_{1,2,3}/file.txt
mv file0001.txt 1.txt
mv file001abc.txt abc1.txt
mv new old -b -S .old
mv new old -b
mv old tmp
mv original.filename new.original.filename
mv {,new.}original.filename
mv svnlog.py svnlog
mv -T www_new www
mv $file $(echo $file | rev | cut -f2- -d- | rev).pkg
find /volume1/uploads -name "*.mkv" -exec rename 's/\.mkv$/.avi/' \{\} \;
find -maxdepth 3 -mindepth 3 -type f -iname '*.jpg' -exec rename -n 's/jpg$/jpeg/i' {} +
mv "$(readlink -f dirln)" dir2
find . -maxdepth 2 -type d | sed 'p;s/thumbs/thumb/' | xargs -n2 mv
find . -type d -exec rename 's/^thumbs$/thumb/' {} ";"
find . -type d | awk -F'/' '{print NF, $0}' | sort -k 1 -n -r | awk '{print $2}' | sed 'p;s/\(.*\)thumbs/\1thumb/' | xargs -n2 mv
find -name '*.html' -print0 | xargs -0 rename 's/\.html$/.var/'
find . -type f -iname '*.txt' -print0 | xargs -0 rename .txt .abc
rename 's/\.html$/\.txt/' *.html
find  | rename 's/\.jpg$/.jpeg/'
rename 's/_h.png/_half.png/' *.png
find . -name "*.txt" | sed "s/\.txt$//" | xargs -i echo mv {}.txt {}.bak | sh
find . -type d -iname '*foo*' -depth -exec rename 's@Foo@Bar@gi' {} +
find . -name CVS -prune -o -exec mv '{}' `echo {} | tr '[A-Z]' '[a-z]'` \; -print
rename -f 'y/A-Z/a-z/' *
rename 'y/A-Z/a-z/' *
rename s/0000/000/ F0000*
rename 's/^fgh/jkl/' fgh*
md5sum * | sed -e 's/\([^ ]*\) \(.*\(\..*\)\)$/mv -v \2 \1\3/e'
find -name 'access.log.*.gz' | sort -Vr | rename 's/(\d+)/$1+1/ge'
find . -type f -inum 31467125 -exec mv {} new_name.html \;
sudo mv edited_blah.tmp /etc/blah
find . -name "*.andnav" -exec rename -v 's/\.andnav$/\.tile/i' {} \;
find . -name "*.andnav" | rename "s/\.andnav$/.tile/"
mv file.txt.123456 $(ls file.txt.123456 | rev | cut -c8- | rev)
find . -type f -inum 31467125 -exec /bin/mv {} new_name.html \;
find ~/junk  -name 'cart1' -exec mv {} ~/junk/A \;
find . -name "article.xml" -exec rename 's/article/001_article/;' '{}' \;
find . -mindepth 2 -maxdepth 2 -name "*.so" -printf "mv '%h/%f' '%h/lib%f'\n" | sh
find . -name "*.so" -printf "mv '%h/%f' '%h/lib%f'\n" | bash
mv $1 `echo $1 | tr '[:upper:]' '[:lower:]'`
echo $(yes image.png | head -n10)
yes image.png | head -n10 | xargs echo
history | sed "s/  / $UID /"
echo "a,b"|sed 's/,/\r\n/'
cat "$file" | sed -e 's/,,/, ,/g' | column -s, -t | less -#5 -N -S
sed 's/,,/, ,/g;s/,,/, ,/g' data.csv | column -s, -t
awk '{gsub(/-/,"0",$4);gsub(/-/,"0",$5)}1' test.in | column -t
echo -e "Testing\r\nTested_Hello_World" | awk -v RS="_" '{ print $0; }' | od -a
echo -e "Testing\r_Tested" | awk -v RS="_" '{ print $0; }' | od -a
sed -i s/'dummyvalue'/$(hostname -I | head -n1 | awk '{print $1;}')/g filename
find -name "*.xml" -exec sed -s --in-place=.bak -e 's/firstWord/newFirstWord/g;s/secondWord/newSecondWord/g;s/thirdWord/newThirdWord/g' {} \;
find . -name "*.php" -exec sed -i 's/foo/bar/g' {} \;
find . -name "*.php" -print | xargs sed -i 's/foo/bar/g'
find . | xargs sed -i ‘s/foo/bar/g’
find . -type f -not -name “.*” -print | xargs sed -i ‘s/foo/bar/g’
sort inputfile | uniq | sort -o inputfile
find ./ -type f -exec sed -i 's/string1/string2/g' {} \;
ln -f -s -T `readlink SomeLibrary | sed 's/version.old/version.new/'` SomeLibrary
find ./ -type f -exec sed -i 's/company/newcompany/' {} \;
find -type f -print0 | xargs -0 sed -i .bakup 's/company/newcompany/g'
tr  ' ' '-'
sudo find . -type f -exec sed -i 's/置換前/置換後/g' {} \;
sed -i.bak "s#https.*\.com#$pub_url#g" MyHTMLFile.html
removestr=$(echo "$list" | tr ":" "\n" | grep -m 1 "^$removepat\$")
sed "s/,/\t/g" filename.csv | less
echo "bla@some.com;john@home.com" | sed -e 's/;/\n/g'
sed -ibak -e s/STRING_TO_REPLACE/REPLACE_WITH/g index.html
sed -i 's/STRING_TO_REPLACE/STRING_TO_REPLACE_IT/g' index.html
sed s/STRING_TO_REPLACE/STRING_TO_REPLACE_IT/g index.html | tee index.html
sed -i.bak s/STRING_TO_REPLACE/STRING_TO_REPLACE_IT/g index.html
sed -i bak -e s/STRING_TO_REPLACE/REPLACE_WITH/g index.html
cat input.txt | sed 's/string/longer_string/g' | column -t
find . -type f -maxdepth 1 -exec sed -i "s/$P_FROM/$P_TO/g" {} \;
awk -F, 'BEGIN {OFS = ","} {gsub("-([0-9.]+)", "(" substr($3, 2) ")", $3); print}' inputfile
sed -i ':a;N;$!ba;s/\n/,/g' test.txt
sed ':a;N;$!ba;s/\n/ /g'
sed 'x;G;1!h;s/\n/ /g;$!d'
sed -e ':a' -e 'N' -e '$!ba' -e 's/\n/ /g'
sed ':a;N;$!ba;s/\n/ /g' file
sed -z 's/\n/ /'g
sed -e '{:q;N;s/\n/ /g;t q}' file
tr -sc '[:punct:]' '\n'
find . -name 'php.ini' -exec sed -i 's/log_errors = Off/log_errors = On/g' "{}" \;
find /home/www -type f -print0 | xargs -0 sed -i 's/subdomainA.example.com/subdomainB.example.com/g'
find /home/www/ -type f -exec sed -i 's/subdomainA\.example.com/subdomainB.example.com/g' {} +
find . \( ! -regex '.*/\..*' \) -type f -print0 | xargs -0 sed -i 's/subdomainA.example.com/subdomainB.example.com/g'
find . -maxdepth 1 -type f -print0 | xargs -0 sed -i 's/toreplace/replaced/g'
find . \( -name "*.php" -or -name "*.html" \) | xargs grep -l StringA | xargs sed -i -e 's/StringA/StringB/g'
find . -name foo_fn exec sed -i s/foo_fn/bar_fn/g '{}' \;
find . -name '*.php' -print0 -o -name '*.xml' -print0 -o -name '*.phtml' -print0 | xargs -0 sed -i '' 's/2013 Magento Inc./2012 Magento Inc./g'
find /home/www -type f -name '*.cpp'  -exec sed -i 's/previousword/newword/g' '{}' \;
find /myprojects -type f -name '*.cpp' -print0 | xargs -0 -n 1 sed -i 's/previousword/newword/g'
find /myprojects -type f -name *.cpp -print0 | xargs -0 sed -i 's/previousword/newword/g'
find /home/www -type f -print0 | xargs -0 sed -i 's/subdomainA.example.com/subdomainB.example.com/g'
find ./cms/djangoapps/contentstore/management/commands/tests -iname *.css | xargs sed -i s/[Ee][Dd][Xx]/gurukul/g
find ./cms/djangoapps/contentstore/views -iname *.css | xargs sed -i s/[Ee][Dd][Xx]/gurukul/g
find %s -iname *.css | xargs sed -i s/[Ff][Oo][Oo]/bar/g
find -name '*.[ch]' -exec sed -i 's/\<foo\>/bar/g' {} +
find ./ -type f -name '*.html' | xargs sed -i 's/<title>/sblmtitle\n<title>/g'
cat log | tr -s [:blank:] |cut -d' ' -f 3-
ARGS="--ignore `echo ${TO_IGNORE[@]} | tr ' ' ','`"
find /tmp/ -depth -name "* *" -execdir rename 's/ /_/g' "{}" \;
find -name "* *" -type d | rename 's/ /_/g'
find -name "* *" -type f | rename 's/ /_/g'
find $DIR -type f -name '*.html' -exec sed -i 's/.*<script type="text\/javascript" charset="utf-8" src="file.js"><\/script>.*/<script type="text\/javascript" charset="utf-8" src="file2.js"><\/script>/g' {} \;
tr '[:blank:]' \\t
cat text.txt | tr -s ' ' | cut -d ' ' -f 4
echo $MOUNT_OPTS | tr , \\\n | grep '^acl$' -q
paste -sd: INPUT.txt
echo "1\n2\n3\n4\n5" | paste -s -d, /dev/stdin
grep -v '^$' YOURFILE | nl -s= -w99 | tr -s ' ' p
grep -o "_foo_" <(paste -sd_ file) | tr -d '_'
find -name "* *" -type d | rename 's/ /_/g'
find -name "* *" -type f | rename 's/ /_/g'
find /tmp/ -depth -name "* *" -execdir rename " " "_" "{}" ";"
find /tmp/ -depth -name "* *" -execdir rename 's/ /_/g' "{}" \;
sudo ln -sf /usr/local/ssl/bin/openssl `which openssl`
find . -type f -name '*.txt' | xargs --replace=FILE sed --in-place 's/foo/baz/' FILE
find ./ -type f -exec sed -i 's/string1/string2/' {} \;
history | sed 's/^ */&\n/; :a; s/ \(.*\n\)/.\1/; ta; s/\n//'
find . -type f -name "*.yaml" -print0 | xargs -0 sed -i -e "s/HOGE/MOGA/"
find . -type f -print0 | xargs -0 sed -i -e "s/HOGE/MOGA/"
find . -type f -print0 | xargs -0 sed -i.bak -e "s/HOGE/MOGA/"
history | sed 's/^\( *[0-9]*\) */\1../'
find . | grep favicon\.ico | xargs -n 1 cp -f /root/favicon.ico
find . -name \*.c -print
find /mydir1 /mydir2 -size +2000 -atime +30 -print
find /mydir1 /mydir2 -size +2000 -atime +30 -print
df -k /tmp | tail -1 | awk '{print $4}'
df -k /tmp | tail -1 | tr -s ' ' | cut -d' ' -f4
df | grep /dev/disk0s2
df -k /example
df -k /tmp
df -h /dir/inner_dir/
df -k /dir/inner_dir/
df -k /some/dir
df -h .
df -k .
df -i $PWD
df -h path-to-file
df .
df -k .
df -h
df -ih
df $path_in_question | grep " $path_in_question$"
df -Ph | column -t
df -Ph
df -BG
df -k
df -P
df --total
df -i
df -i
df --total -BT | tail -n 1 | sed -E 's/total *([^ ]*).*/\1/'
df -m | awk '{ SUM += $2} END { print SUM/1024/1024"TB" }'
df -h /
df -H --total /
df --total -BT | tail -n 1
df --total -BT | tail -n 1
df --total | tail -n 1
df -H --total /
echo -e "length(FOO_NO_EXTERNAL_SPACE)==$(echo -ne "${FOO_NO_EXTERNAL_SPACE}" | wc -m)"
echo -e "length(FOO_NO_LEAD_SPACE)==$(echo -ne "${FOO_NO_LEAD_SPACE}" | wc -m)"
echo -e "length(FOO_NO_TRAIL_SPACE)==$(echo -ne "${FOO_NO_TRAIL_SPACE}" | wc -m)"
echo -e "length(FOO_NO_WHITESPACE)==$(echo -ne "${FOO_NO_WHITESPACE}" | wc -m)"
echo -e "length(FOO)==$(echo -ne "${FOO}" | wc -m)"
grep "^core id" /proc/cpuinfo | sort -u | wc -l
grep '^core id' /proc/cpuinfo |sort -u|wc -l
timestamp=`date --rfc-3339=seconds`
t1=$(date -u -d "1970.01.01-$string1" +"%s")
TODAY=$(date  -d "$(date +%F)" +%s)
MOD_DATE1=$(date -d "$MOD_DATE" +%s)
dig @$ns $d A | grep $d | grep -v "DiG"
dig +trace +additional dragon-architect.com | awk 'NR==3' RS="\n\n"
dig +short -f - | uniq
dig +short myip.opendns.com @resolver1.opendns.com
dig mx example.com | grep -v '^;' | grep example.com
dig NS +aaonly com.
dig @ns1.hosangit.com djzah.com +noall +authority +comments | awk -f script.awk
ssh-keygen -pf private.key
ssh -O exit officefirewall
ssh -O exit otherHosttunnel
source ~/.profile
readlink -f $(which firefox)
cd "`pwd -P`"
readlink -e /foo/bar/baz
readlink -m FILE
readlink $(which python2.7)
head -1 table | tr -s ' ' '\n' | nl -nln |  grep "Target" | cut -f1
uname -v | grep -o '#[0-9]\+'
find . -name 'abc' -type f -exec grep -q xyz {} +
df /full/path | grep -q /full/path
find . -mnewer poop
find /home/pat -iname "*.conf"
find . -type d -printf '%d:%p\n' | sort -n | tail -1
find / -newer myfile
find / -size +50M -iname "filename"
groups $1 | grep -q "\b$2\b"
a=$(false)
a=$(false)
false
false | echo "${PIPESTATUS[0]}"
foo=$(false)$(true)
ls -t | head -n1
find . -name '*tests*' -print -exec false \;
cat dax-weekly.csv | awk '1 { last = NR; line[last] = $0; } END { print line[1]; for (i = last; i > 1; i--) { print line[i]; } }'
tail -r myfile.txt
echo 35 53 102 342|tr ' ' '\n'|tac|tr '\n' ' '
echo "aaaa eeee bbbb ffff cccc"|tr ' ' '\n'|tac|tr '\n' ' '
output=$(echo $input | fold -w4 | tac | tr -d \\n)
cat ${TMP}/${SCRIPT_NAME}.kb|sort -rh;
cat ${TMP}/${SCRIPT_NAME}.name|sort -r;
cat ${TMP}/${SCRIPT_NAME}.pid|sort -rh;
echo $string | rev | cut -d ' ' -f -20
pushd -2
CC=$(which cc) ./configure
CC=$(which gcc) ./configure
echo "command" | ssh user@host
find . -type f -exec chmod 0644 {} \;
find . -type d -exec chmod 0755 {} \;
join -t $'\t' file1 file2
join <(sort -n file1) <(sort -n file2)
HOSTNAME=$(hostname) make -e
env `cat xxxx` otherscript.sh
env -u FOO somecommand
top -b -d 1 | grep myprocess.exe | tee output.log
env -i ./makeall.sh
nohup rm -rf cache &
env - `cat ~/cronenv` /bin/sh
find . -name "*.txt" -print -exec awk '$9 != "" && n < 10 {print; n++}' {} \;
find . -name "*.txt" -print -exec awk '$9 != "" {print; if(NR > 9) exit; }' {} \;
find . -name "*.txt" -print -exec awk '{if($9!=""&&n<11){print;n++}}' {} \;
find . -name '*txt' -print -exec awk 'BEGIN {nl=1 ;print FILENAME} $9 !="" {if (nl<11) { print $0 ; nl = nl + 1 }}' {}  \;
find ./ -type f -print0 | xargs -0 -n1 md5sum | sort -k 1,32 | uniq -w 32 -d --all-repeated=separate | sed -e 's/^[0-9a-f]*\ *//;'
echo "su whoami" |ssh remotehost
arr=$( $line | tr " " "\n")
echo "df -k;uname -a" | ssh 192.168.79.134
env -i perl -V
rsync $OPTS $FIND $BACKUPDIR
cat $2 | grep -v "#" | ssh -t $1 $INTERPRETER
sed -i "s#\(export\ PATH=\"\)\(.*\)#\1/home/$(whoami)/bin:~/\.local/bin:\2#" ~/.zshrc
find . -type f -exec file '{}' \;
`which find` "$@" -print0;
find . -type f -exec file '{}' \;
ssh -x user@server
ssh -o "StrictHostKeyChecking no" -i ${KEYDIR}/${KEY}.pem ${USERNAME}@$NAME "${COMMANDS}"
ssh -l myName -p 22 hostname
ssh -p 22 myName@hostname
REL_DIR="$(ssh -t localhost "$heredoc")"
ssh -o ServerAliveInterval=60 myname@myhost.com
ssh "$1" -l pete
ssh -i /path/to/ssh/secret/key $1 $2
ssh user@server
ssh app1
ssh remote_user@server.com
ssh user@server "${SSH_COMMAND}"
ssh -i ./device_id.pem deviceuser@middle.example.org:2222
ssh -o ControlPath="$MASTERSOCK" -MNf "$@"
ssh -i ~/.ssh/gitkey_rsa "$@"
ssh -XY -t user@remote_IP 'ssh -XY -t user@remoteToRemote_IP'
myvar=`seq 1 $N | sed 's/.*/./' | tr -d '\n'`
echo something | read param
history -s "$line"
history -s 'echo whatever you "want your" command to be'
echo foo | read bar
WORKSTATION=`who -m|awk '{print $5}'|sed 's/[()]//g'`
ip=$(hostname -I | awk '{print $1}')
ip=$(hostname -I)
JAVA_HOME="$( readlink -f "$( which java )" | sed "s:bin/.*$::" )"
FOLDERS=`ls -dm $MY_DIRECTORY/*/ | tr -d ' '`
FOLDERS=$(find . -type d -print0 | tr '\0' ',')
foo=$(cat /dev/urandom | tr -dc '. ' | fold -w 100 | head -1)
files="$(find $dir -perm 755)"
find . -fprint foo
path="http://$(whoami).$(hostname -f)/path/to/file"
RAND=`od -t uI -N 4 /dev/urandom | awk '{print $2}'`
bgxjobs=" $(jobs -pr | tr '\n' ' ')"
line=$(who | cut -d' ' -f1 | sort -u)
abspath=$(readlink -m $path)
abspath=$(readlink -e $path)
abspath=$(readlink -f $path)
absolute_path=$(readlink -m /home/nohsib/dvc/../bop)
MY_PATH=$(readlink -f "$0")
SCRIPT="$(readlink --canonicalize-existing "$0")"
target_PWD=$(readlink -f .)
FOLDERS=$(find $PWD -type d | paste -d, -s)
FOLDERS=$(find . -type d | paste -d, -s)
var=`egrep -o '\[.*\]' FILENAME | tr -d ][`
totalLineCnt=$(cat "$file" | grep "$filter" | grep -v "$nfilter" | wc -l | grep -o '^[0-9]\+');
libdir=$(dirname $(dirname $(which gcc)))/lib
address=$(dig +short google.com | grep -E '^[0-9.]+$' | head -n 1)
CURRENT_PID_FROM_LOCKFILE=`cat $LOCKFILE | cut -f 1 -d " "`
CAT=`which cat`
MKTEMP=`which mktemp`
RM=`which rm`
TR=`which tr`
day=$(od -t x1 --skip-bytes=9 --read-bytes=1 file.moi | head -1 | awk '{print $2}')
month=$(od -t x1 --skip-bytes=8 --read-bytes=1 file.moi | head -1 | awk '{print $2}')
year=$(od -t x2 --skip-bytes=6 --read-bytes=2 file.moi | head -1 | awk '{print $2}')
thisHOSTNAME=`hostname`
QUEUE_PIDS=$(comm -23 <(echo "$NEW_PIDS" | sort -u) <(echo "$LIMITED_PIDS" | sort -u) | grep -v '^$')
results=$(groups "$line" | tr ' ' '\n' | egrep -v "_unknown|sciences|everyone|netaccounts")
listing=$(ls -l $(cat filenames.txt))
cnt=`ps -ef| tee log | grep "cntps"|grep -v "grep" | wc -l`
LINES=$(cat /some/big/file | wc -l)
nbLines=$(cat -n file.txt | tail -n 1 | cut -f1 | xargs)
MERGE=$(cat $COMMIT_EDITMSG|grep -i 'merge'|wc -l)
NP=`cat /proc/cpuinfo | grep processor | wc -l`
big_lines=`cat foo.txt | grep -c "$expression"`
number=$(echo $filename | tr -cd '[[:digit:]]')
DayOfWeek=`date +%a |tr A-Z a-z`
myVar=$(tee)
read -d "$(echo -e '\004')" stdin
filename="$(uname -a)$(date)"
pushd /home/`whoami`/Pictures
fhost=`hostname -f`
hnd=$(hostname -f)
FinalDate=$(date -u -d "$string2" +"%s")
full_path=`readlink -fn -- $path`
fullpath=`readlink -f "$path"`
SELF=$(readlink /proc/$$/fd/255)
SELF=`readlink /proc/$$/fd/255`
actual_path=$(readlink -f "${BASH_SOURCE[0]}")
script="`readlink -f "${BASH_SOURCE[0]}"`"
DIR=$(dirname "$(readlink -f \"$0\")")
me=$(readlink --canonicalize --no-newline $BASH_SOURCE)
me=$(readlink --canonicalize --no-newline $0)
path=`readlink --canonicalize "$dir/$file"`
JAVA_HOME=$(readlink -f /usr/bin/java | sed "s:/bin/java::")
f=$(cat numbers.txt)
DATE=$(echo `date`)
subj="$(date) - $(hostname) - $(echo "$changes" | sed "s/$/,/" | tr "\n" " ")"
CDATE=$(date "+%Y-%m-%d %H:%M:%S")
myvariable=$(whoami)
DIR=`pwd`/`dirname $0`
CURRENT=`pwd`
real1=$(pwd -P)
date_222days_before_TodayDay=$(date --date="222 days ago" +"%d")
DATECOMING=$(echo `date -d "20131220" +%j`)
MY_DIR=$(dirname $(readlink -f $0))
path="$( dirname "$( which "$0" )" )"
dir=$(dirname $(readlink -m $BASH_SOURCE))
dir=$(dirname $(readlink /proc/$$/fd/255))
HOSTZ=$( hostname | cut -d. -f1 )
yes | awk 'FNR<4 {print >>"file"; close("file")}  1' | more
echo $j | read k
local=$(hostname -I | awk '{print $2}' | cut -f1,2,3 -d".")
subnet=$(hostname -i | cut -d. -f1,2,3)
DC=`hostname | cut -b1,2`
extract_dir=$(diff .dir_list_1 .dir_list_2 | grep '>' | head -1 | cut -d' ' -f2)
full_f="$(which f)"
path=`which oracle`
foo=`which ~/f`
ver=`echo -ne "$1\n$2" |sort -Vr |head -n1`
STAMP=`date -r file_name`
timestamp=$(find ./$dir -type f -printf "%T@ %t\\n" | sort -nr -k 1,2 | head -n 1)
line_to_be_replaced=`cat itemlist.json | nl |  sed -n '/"item_1"/,/"item_2"/p' | grep -in "}]" | awk '{print $2}'`
find . -name '*.py' | tee output.txt | xargs grep 'something'
basedir=$(pwd -L)
md5=$(md5sum "$my_iso_file" | cut -d ' ' -f 1)
md5=`md5sum ${my_iso_file} | awk '{ print $1 }'`
twofish=`echo -n $twofish | md5sum | tr -d "  -"`
filename="$(uname -n)-$(date +%F).txt"
size="$(zcat "$file" | wc -c)"
candidates=$(which -a $cmd | wc -l)
server_id=`hostname | tr 'A-Za-z-.' ' ' | tr -d '[[:space:]]' | awk '{print NR}'`
gv=$(echo -e $kf'\n'$mp | sort -t'.' -g | tail -n 1)
packet_loss=$(ping -c 5 -q $host | grep -oP '\d+(?=% packet loss)')
END_ABS=`pwd -P`
WORKSTATION_IP=`dig +short $WORKSTATION`
DBPREFIX="$(hostname -s).mysqldb"
hostname=`hostname -s`
HOSTNAME="`hostname`"
HOSTNAME=$(hostname)
HOST=$(hostname)
myHostName=`hostname`
proc_load_average=$(w | head -1 | cut -d" " -f12 | cut -d"," -f1-2 | tr ',' '.')
v=$(whoami | awk '{print toupper($0)}')
v=$(whoami | tr 'a-z' 'A-Z')
v=$(whoami | tr [:lower:] [:upper:])
me="$(whoami)"
me=$(whoami)
whoami=$(whoami)
x=$(whoami)
tmux_version="$(tmux -V | cut -c 6-)"
var2=$(echo $myvar | wc -c)
uiTotalSize=$(ls -l -R $1 | grep -v ^d | awk '{total+=$5;} END {print total;}')
a=$(echo $each | wc -c)
cal=$(echo $(cal "$month" "$year"))
do=$(cal -m $mo $yo|awk 'NR>2&&!/^  /{print$1;exit}')
true | true | false | true | false
true | false | true
false | true
size=`cat script.sh | wc -c`
base=$(dirname $(readlink $file))
host=$(dig +short -x "${ip_address}" | sed 's/\.$//g')
result=$(groups "$line" | sed 's/ /\n/g' | egrep -v "_unknown|sciences|everyone|netaccounts")
inode=`ls -i ./script.sh | cut -d" " -f1`
check_script_call=$(history |tail -1|grep myscript.sh )
userlist=$(w|awk 'BEGIN{ORS=","}NR>2{print $1}'|sed 's/,$//' )
a=`w|cut -d' ' -f1`;
b=`w|cut -d' ' -f1`;
OUTPUT="$(ls -1)"
var=$(ls -l)
dir=$(dirname -- "$1")
dir_context=$(dirname -- "$1")
tmp=$(w | awk '{print $1}')
n_max=`ls . | wc -l`
set SCRIPTPATH=`dirname "$SCRIPT"`
set `cal $month $year`
set -- $(cal 2 1900)
DIR=$(dirname "$(readlink -f \"$0\")")
FILES=`cat $RAW_LOG_DIR | xargs -r`
false | tee /dev/null
find /etc -type f -exec cat '{}' \; | tr -c '.[:digit:]' '\n' | grep '^[^.][^.]*\.[^.][^.]*\.[^.][^.]*\.[^.][^.]*$'
find /etc -exec grep '[0-9][0-9]*[.][0-9][0-9]*[.][0-9][0-9]*[.][0-9][0-9]*' {} \;
find /usr -inum 1234 -print
find /  -size +100M -exec rm -rf {} \;
find . -perm 777 -print
find . -atime +7 -print
find . \( -user aa1 -o -name myfile \) -print
find .  -size 10 print
find . -size +10c -print
grep -f file2 file1 | sort -u
grep -f file2 file1 | sort | uniq
grep -of ignore.txt input.txt | sort | uniq -c
cat inputfile | grep "^t\:" | split -l 200
gunzip -c mygzfile.gz | grep "string to be searched"
history | grep -A 4 -B 3 whatyousearchfor
MYUSERS=`grep $MYGROUP /etc/group | cut -d ":" -f4| tr "," "\n"`
find /directory/containing/files -type f -name "*.txt" -exec grep -H 'pattern_to_search' {} +
find . -name ‘*.x’ -print0 | xargs -0 grep fred
find . -name '*.[ch]' | xargs grep -E 'expr'
find /dev/shm /tmp -type f -ctime +14
find /etc -type f -mmin -10
find /etc -type f -ctime -1
find /path/to/your/directory -regex '.*\.\(avi\|flv\)'
find /public/html/cosi -name "wiki.phtml"
find /root/directory/to/search -name 'filename.*'
find /some/directory -user joebob -print
find /tmp -size -100c
find /usr /home  /tmp -name "*.jar"
find /usr/bin -type f -mtime -10
find /usr/bin -type f -atime +100
find /usr/local -maxdepth 1 -type d -name '*[0-9]'
find /usr/local -type d -name '*[0-9]'
find /usr/src ! \( -name '*,v' -o -name '.*,v' \) '{}' \; -print
find /var -regex '.*/tmp/.*[0-9]*.file'
find /var/log -size +10M -ls
find /tmp /var/tmp -size +30M -mtime 31 -ls
which -a rename | xargs file -L
which -a rename | xargs readlink -f | xargs file
find . -name \*.c -exec grep hogehoge {} \;
find . -name \*.c -print | xargs grep hogehoge
find . -name \*.c -print0 | xargs -0 grep hogehoge /dev/null
find ~/documents -type f -name '*.txt' -exec grep -s DOGS {} \; -print
find . -name "*.[ch]" -exec grep --color -aHn "e" {} \;
find . -name "*.c" -exec grep -i "keyword" {} ";"
find . -name '*.java' -mtime +7 -print0 | xargs -0 grep 'swt'
find . -name \*.py | xargs grep some_function
find . -name "*.py" | xargs grep 'import antigravity'
find / -iname "filename"
find /home/oracle /home/database -name '*zip*'
find . -name "abc" -exec grep "xyz" {} \;
find . -name abc | xargs grep xyz
find /tmp -type f -exec grep 'search string' '{}' /dev/null \+
find . \( -name '*.svn*' -prune  -o ! -name '*.html' \) | xargs -d '\n' grep -Hd skip 'SearchString'
find . -name whatever -print | xargs grep whatever
find -name whatever -exec grep --with-filename you_search_for_it {} \;
find . -name '*.*' -exec grep 'SearchString' {} /dev/null \;
find . -name "*1" -exec grep "1" {} +
find . -name "*1" -exec grep "1" {} \;
find . -name "*1" -print0 |xargs -0 grep "1"
find . ! -name '*.html' ! -name '*.svn*' -exec grep 'SearchString' {} /dev/null \;
find . -name .git -prune -o -print | xargs grep "string-to-search"
find /etc -exec grep '[0-9][0-9]*[.][0-9][0-9]*[.][0-9][0-9]*[.][0-9][0-9]*' {} \;
find *-name hi.dat
find /usr -type d -name 'My Files' -exec rsync -avR '{}' /iscsi \;  -exec rm -rf '{}'\;
find /usr -type d -name My\ Files -exec rsync -avR '{}' /iscsi \;
find -maxdepth 1 -type f | xargs grep -F 'example'
find -type f -print0 | xargs -r0 grep -F 'example'
find . -type f -exec grep string {} \;
find . -type f | xargs -d '\n' grep string
find . -name '*.pl' | xargs grep -L '^use strict'
find -name '*.[ch]' | xargs grep -E 'expr'
find . -type f -print -exec grep --color=auto --no-messages -nH "search string" "{}" \;
find -maxdepth 1 -type f | xargs grep -F 'example'
find -type f -print0 | xargs -r0 grep -F 'example'
env | grep NAME
tail -f logfile.log | grep --line-buffered "something" | read -t 3
grep -ioh "facebook\|xing\|linkedin\|googleplus" access-log.txt | sort | uniq -c | sort -n
find . -type f -name "*.java" -exec grep -il 'foo' {} \;
find . -name core -exec rm {} \;
cd "$(find . -name config -type d | sed 1q)"
zcat filename.gz | grep -i user-user
find /opt /usr /var -name foo -type f
find /res/values-en-rUS /res/xml -iname '*.xml'
find /usr -type d -name backup -print
find ${CURR_DIR} -type f \( -ctime ${FTIME} -o -atime ${FTIME} -o -mtime ${FTIME} \) -printf "./%P\n"
find /Users/david/Desktop/ -type f
find /Users/david/Desktop -type f \( -name '*.txt' -o -name '*.mpg' -o -name '*.jpg' \)
find /home/ABCD/ -mindepth 1 -type f -print
find /path/to/check/* -maxdepth 0 -type f
find /tmp/foo -path /tmp/foo/bar -print
find foo -path /tmp/foo/bar -print
find foo -path foo/bar -print
find lpi104-6 -inum 1988884
find "$DIR" -name \*.txt
find "${DIR}" -name "*.txt"
find $DIR -name "*.txt"
find $DIR -name "*.txt" -print
find /srv/${x} -mindepth 1 -type f -not -amin -10080 -exec rm {} \;
find -d MyApp.app -name Headers -type d -exec rm -rf "{}" \;
find MyApp.app -name Headers -type d -delete
find MyApp.app -name Headers -type d -exec rm -rf "{}" \;
find -d MyApp.app -name Headers -type d -exec rm -rf {} +
find -d MyApp.app -name Headers -type d -print0 | xargs -0 rm -rf
find foo -name Headers
find /tmp /var/tmp -iname "testfile.txt"
find /usr/local/man /opt/local/man -name 'my*'
find /usr/share/doc /usr/doc /usr/locale/doc -name instr.txt
find foo bar -name "*.java"
find / -path excluded_path -prune -o -type f -name myfile -print
find / -name httpd.conf -newer /etc/apache-perl/httpd.conf
find / -type d -name root
find / -newerct '1 minute ago' -print
find / -name .profile
find /etc/logs/Server.log -exec grep Error {} \; -print
find . -name aaa -print
find . -name "file-containing-can't" -exec grep "can't" '{}' \; -print
find /home/ABCD/ -type f -print
od file-with-nulls | grep ' 000'
history | grep " 840"
find . -maxdepth 1 ! -regex '.*~$' ! -regex '.*\.orig$' \     ! -regex '.*\.bak$' -exec grep --color "#define" {} +
find . -newermt '2014-04-30 08:00:00' -not -newermt '2014-04-30 09:00:00' |xargs gunzip -c | grep 1234567890
zcat /proc/config.gz | grep CONFIG_64BIT
find ~jsmith -exec grep LOG '{}' /dev/null \; -print
find . -name "*.java" | xargs grep "Stock"
grep YOURSTRING `find .`
grep -ioh "facebook\|xing\|linkedin\|googleplus" access-log.txt | sort | uniq -c | sort -n
grep foo * | nl
history | grep -C 5 ifconfig
history | grep ifconfig -A5 -B5
ps -u `whoami` | grep largecalculation
cat file | grep pattern | paste -sd' '
cat file | grep pattern | paste -sd'~' | sed -e 's/~/" "/g'
find . -name "*.c" | xargs grep pattern
find . | awk '{FS = "/" ; print "", NF, $F}' | sort -n  | awk '{print $2}' | xargs grep -d skip "search term"
zcat `find /my_home -name '*log.20140226*'`|grep 'vid=123'
find /dir -type f -print0 | xargs -0i cat {} | grep whatever
find . -exec grep "www.athabasca" '{}' \; -print
grep ^$GROUP /etc/group | grep -o '[^:]*$' | tr ',' '\n'
find $DIR -type f -exec grep $SEARCH /dev/null {} \; | wc --bytes
find . -type f -exec grep "/usr/bin/perl" {} \; -ls
find /proc/scsi/ -path '/proc/scsi/usb-storage*' -type f | xargs grep -l 'Attached: Yes'
find ~/documents -type f -name '*.txt' -exec grep -s DOGS {} \; -print
find . -iname "$srch1*" -exec grep "It took" {} \; -print
find . -iname "$srch1*" -exec grep "It took" {} \; -print |sed -r 'N;s/(.*)\n(.*)/\2 \1/'
find . -iname "$srch*" -exec grep "Processed Files" {} \; -print
find . -iname "$srch*" -exec grep "Processed Files" {} \; -print| sed -r 'N;s/(.*)\n(.*)/\2 \1/'
find . -name * -print0  | xargs -0 grep -iH "String"
find ./ -type f -exec grep -l "Text To Find" {} \;
find ./ -type f -exec grep -Hn "Text To Find" {} \;
find ~/Documents -type f -print0 | xargs -0 grep -il birthday
find -iname '*.java'|xargs grep 'class Pool'
find -maxdepth 1 -type f | xargs grep -F 'example'
find -type f -print0 | xargs -r0 grep -F 'example'
find . -type f -print | xargs grep "example"
find . -type f -exec grep "example" '{}' \; -print
find sources -type f -exec grep -H foo {} +
find dir1 dir2 dir3 -type f -name "*.java" -exec grep -il 'foo' {} \;
find . -name "*.png" -prune -o -name "*.gif" -prune -o -name "*.svn" -prune -o -print0 | xargs -0 -I FILES grep -IR "foo=" FILES
find ./online_admin/*/UTF-8/* -type f -exec grep -H "invalidTemplateName" {} \;
find project -name '*.php' -type f -print0 | xargs -0 grep -l ireg
find . | grep 'some string' | grep js
find ./ -not -path "*/node_modules/*" -name "*.js" | xargs grep keyword
find . -type f | xargs grep "magic"
find . -iname *.txt -exec egrep mystring \{\} \;
find . -name "*.txt" -exec egrep mystring {} \;
find . -name "*.txt" -print0 | xargs -0 egrep mystring
find . -name '*.txt' | xargs egrep mystring
find . -name *.txt | xargs egrep mystring
find ~/html/ -name '*.html' -exec grep organic '{}' ';'
find . -name “*.cc” |xargs grep -n “pattern”
grep pattern file | tr '\n' ' '
find /directory/containing/files -type f -name "*.txt" -exec grep -H 'pattern_to_search' {} +
find . -name '*.axvw' -exec grep -n 'some string' {} +
find . -name '*.axvw' -print0 | xargs -0 grep -n 'some string'
find . -name '*js' -exec grep -n 'some string' {} \;
find . -name '*js' | grep -n  'some string'
find . -name '*.txt' -exec grep 'sometext' '{}' \; -print
find . -name “*.[txt|TXT]” -print | xargs grep “specific string”
find . -name * | xargs grep -iH "string"
find . -name \*.html -exec grep -H string-to-find {} \;
find . -exec grep -H string-to-find {} \;
find . -regex filename-regex.\*\.html -exec grep -H string-to-find {} \;
find -type f | sed 's/./\\&/g' | xargs grep string_to_find
find . -name "*.txt" -print0 | xargs -0 egrep 'stuff'
find . -type f -exec grep "text" {} /dev/null \;
find . -exec grep whatIWantToFind {} \;
find / -name ‘*.*’ -exec grep -il “foobar” {} \;
find -name "*pattern*"
find . -name "*.bam"
find / -name '*.pdf'
find / -type f -name *.zip -size +100M -exec rm -i {} \;
grep ERROR $(find . -type f -name 'btree*.c')
cd "$(find . -name Subscription.java -printf '%h\n')"
find . -name '*.pl' | xargs grep -L '^use strict'
find . -name "file-containing-can't" -exec grep "can't" '{}' \; -print
find . -type f -exec grep -o aaa {} \; | wc -l
find / -type f -name "*.conf"
find $HOME -iname '*.ogg' -atime +30
find www -name \*.html -type f -exec basename {} \;
find /mnt/usb -name  "*.mp3" -print
find . -iname foo -type d
find . -iname foo
find /home -xdev -inum 2655341
find . -name "Linkin Park*"
find . -iname "*linkin park*"
find . -iname *linkin*
find . -name "*Linkin Park"
find . -iname foo -type f
find / -newer /tmp/t
find / -newer /tmp/t1 -and -not -newer /tmp/t2
find / -not -newer /tmp/t
find -user www-data -not -size +100k
find / -type f -name "*.conf"
find . -ipath '*sitesearch*' -ipath '*demo*'
find . -iregex '.*sitesearch.*' -iregex '.*demo.*'
find . | grep -i demo | grep -i sitesearch
find . -inum NUM
find . -follow -inum 41525360
find / -name *.jpg -type f -print | xargs tar -cvzf images.tar.gz
find . -name '*'
find /path/to/folders/* -type d -exec mv {} {}.mbox \; -exec mkdir {}.mbox/Messages \;
find . -name 'm?' -type d -exec mv '{}' '{}.mbox' ';' -exec mkdir '{}.mbox/Messages' ';' -exec sh -c 'mv {}.mbox/*.emlx {}.mbox/Messages' ';'
find /usr/share/man/ -regex .*/grep*
find test -type f  -size 0 -exec mv {} /tmp/zerobyte \;
find -L -type l
find . -name "*.[!r]*" -exec grep -i -l "search for me" {} \;
zcat log.tar.gz | grep -a -i "string"
find . -name '*foo*' ! -name '*.bar' -type d -print
find . -iname foo -type d
find . -size 0k
set | grep HIST
find -mindepth 2 -maxdepth 3 -name file
find -mindepth 4 -name file
find . -name '*[+{;"\\=?~()<>&*|$ ]*' -maxdepth 0 -exec rm -f '{}' \;
ls | grep android | nl
find . -name file1 -or -name file9
find ~ -size +10M
find . -exec grep PENWIDTH {} \; | more
find / -type f -size +20M -exec ls -lh {} \; | awk '{ print $NF ": " $5 }'
find ~ -size +20M
find ~/ -atime +100 -exec rm -i {} \;
find $HOME  -mtime 0
find $HOME  -mtime 0
find . -name \*.php -type f -exec grep -Hn '$test' {} \+
find . -name \*.php -type f -exec grep -Hn '$test' {} \;
find . -name \*.php -type f -print0 | xargs -0 -n1 grep -Hn '$test'
find . -name \*.php -type f -print0 | xargs -0 grep -Hn '$test'
find -maxdepth num -name query
find -mindepth num -name query
find / -size +1.1G
find / -size +100M
find -atime -5
find . -perm /222
find . -perm -664
find . -perm 664
find / -perm 777 -iname "filename"
find . -size +100k -a -size -500k
find . -iname '*demo*' | grep -i sitesearch
find . -iname '*sitesearch*' | grep demo
find /path/to/folder -path "*/ignored_directory" -prune -o -name fileName.txt -print
find . -path ./ignored_directory -prune -o -name fileName.txt -print
find . -perm -444 -perm /222 ! -perm /111
find . -perm -220
find . -perm -g+w,u+w
find . -perm /220
find . -perm /u+w,g+w
find . -perm /u=w,g=w
find . -perm /222
find . -perm -664
find . -perm 664
find / -iname '*.txt'
find /var/log/ -iname anaconda*
find /var/log/ -iname anaconda.*
find /var/log/ -iname anaconda.* -exec tar -cvf file.tar {} \;
find var/log/ -iname anaconda.*
find var/log/ -iname "anaconda.*" -exec tar -rvf file.tar {} \;
find var/log/ -iname anaconda.* -exec tar -cvf file.tar {} \;
find var/log/ -iname anaconda.* | xargs tar -cvf file1.tar
tar -cvf file.tar `find var/log/ -iname "anaconda.*"`
find var/log -print0 -iname 'anaconda.*' | tar -cvf somefile.tar -T -
find . -name '*.txt'|xargs grep -m1 -ri 'oyss'
find . -name \*.coffee -exec grep -m1 -i 're' {} \;
find . -print0 -name '*.coffee'|xargs -0 grep -m1 -ri 're'
find . -name '*.coffee' -exec awk '/re/ {print;exit}' {} \;
find . -name \*.coffee -exec awk '/re/ {print;exit}' {} \;
find . -name \*.coffee -exec awk '/re/ {print FILENAME ":" $0;exit}' {} \;
find . -name '.?*' -prune
nl -ba  -nln  active_record.rb  | grep -C 2 '^111 '
nl -ba  -nln  active_record.rb  | grep '^111 '
grep '^[[:space:]]*http://' | sort -u | nl
find . ! -size 0k
find . -name "*.c" -print | xargs grep "main("
find . -type f -name "*.c" -print -exec grep -s "main(" {} \;
find -name '*.[ch]' | xargs grep -E 'expr'
find . -iname foo -type f
find /  -type f -group users
find / -type f -user bluher -exec ls -ls {}  \;
find / -type l -lname '/mnt/oldname*'
find . -name *.xml | xargs grep -P "[\x80-\xFF]"
find . -print | xargs grep -l -i "PATTERN"
find . \( \( -name .svn -o -name pdv \) -type d -prune \) -o \( -name '*.[pwi]' -type f -exec grep -i -l "search for me" {} + \)
find . -type f -exec grep -n -i STRING_TO_SEARCH_FOR /dev/null {} \;
find . -name "$1" -type f -exec grep -i "$2" '{}' \;
find . -name "$1" -type f -print0 | xargs -0 grep -i "$2"
find . -name $1 -type f -exec grep -i $2 '{}' \;
find . -name '*.[ch]' | xargs grep -E 'expr'
find /var/log/apache2/access*.gz -type f -newer ./tmpoldfile ! -newer ./tmpnewfile \ | xargs zcat | grep -E "$MONTH\/$YEAR.*GET.*ad=$ADVERTISER HTTP\/1" -c
find . -name '*.[ch]' | xargs grep -E 'expr'
find -user root -o -user www-data
find . -mtime +1
find . -print|xargs grep v\$process
find . -name '*.pl' | xargs    grep -L '^use strict'
find /tmp -type f -exec grep 'search string' '{}' /dev/null \+
find -iname "filename"
grep -o "+\S\+" in.txt | tr '\n' ','
find /home/*/public_html/ -type f -iwholename "*/wp-includes/version.php" -exec grep -H "\$wp_version =" {} \;
find /var/www/vhosts/*/httpdocs -type f -iwholename "*/wp-includes/version.php" -exec grep -H "\$wp_version =" {} \;
find "$searchpath" -name "$filepat.[ch]" -exec grep --color -aHn "$greppattern" {} \;
find /users/tom -name '*.p[lm]' -exec grep -l -- '->get(\|#hyphenate' {} +
find /etc -exec grep '[0-9][0-9]*[.][0-9][0-9]*[.][0-9][0-9]*[.][0-9][0-9]*' {} \;
find . -name "*.log" -exec egrep -l '^ERROR' {} \;
find /directory/containing/files -type f -exec grep -H 'pattern_to_search' {} +
man find | grep ...
find -type f | xargs head -v -n 1 | grep -B 1 -A 1 -e '^catalina'
find "$DIR" -type f -exec grep -q "$SEARCH" {} + ;
find "$DIR" -type f -exec grep "$SEARCH" {} \;
find . -name "*.xml" -exec grep "ERROR" /dev/null '{}' \+
find / -type f -print | xargs grep "device"
grep foo `find /usr/src/linux -name "*.html"`
find . -not -path "*git*" -not -name '*git*' |grep git
find . -path ./.git -prune -o -not -name '*git*' -print |grep git
find . -type f -exec grep "magic" {} \; -ls
more /etc/hosts | grep `hostname` | awk '{print $1}'
find /var/www/ -name wp-config.php -maxdepth 2
file $(which foo)
history | grep 'part_of_the_command_i_still_remember_here'
find .  \( -user aa1 - group grp \) -print
find . -size 10c -print
$ find . \( -name D -prune \) -o -name hi.dat
find -path ./D -prune -o -name hi.dat -print
find . \( -name D -prune \) -o -name hi.dat
find -mindepth 3 -maxdepth 3 -type d | grep "New Parts"
find my_folder -type f -exec grep -l "needle text" {} \; -exec file {} \; | grep text
find MyApp.app -name Headers -type d -prune -exec rm -rf {} +
find -type d
find -type f
find /mydir | xargs -I{} basename {}
find /path/to/dir -type f -exec sed '/@GROUP/,/@END_GROUP/!d' {} + | grep '_START'
find /path/to/dir -type f -exec sed '/@GROUP/,/@END_GROUP/!d' {} \; | grep '_START'
find ~/mail -type f | xargs grep "Linux"
find -path './Linux/*' -name teste.tex
find katalogi -name wzorzec
find test1 -type f -print
find test1 -type f -name 'textfile.txt' -print
find your/dir -mindepth 1 -prune -empty
find tmp -maxdepth 1 -name '*.mp3'
find . -name *.c -exec grep -n -e blash {} \;
find . -name "*.cc" -print -exec grep "xxx" {} \;
find . -name "*.cc" | xargs grep "xxx"
find . -name '*.code' -exec grep -H 'pattern' {} +
find . -name '*.code' -print0 | xargs -0 grep -H 'pattern'
find . -name "*.txt" -print0 | xargs -0 egrep 'string'
find ./bin -name “cp”
find ./in_save/ -type f -maxdepth 1| more
find . -name "*.VER" -exec grep -P 'Model-Manufacturer:.\n.' '{}' ';' -print
find . -name "*.VER" -exec grep 'Test_Version=' '{}' ';' -print;
find Lib/ -name '*.c' -print0 | xargs -0 grep ^PyErr
find ./ -type f -iname "*.cs" -print0 | xargs -0 grep "content pattern"
find /starting/directory -type f -name '*.css' | xargs -ti grep '\.ExampleClass' {}
find /Applications/ -name "*.java" -exec grep -i TODO {} +
find /Applications/ -name "*.java" -exec grep -i TODO {} \;
find /Applications/ -name "*.java" -print0 | xargs -0 grep -i "TODO"
find . -name "*.java" -exec grep -Hin TODO {} \;
find . -name "*.java" -exec grep -i -n TODO {}  \;
find `pwd` -name "*.log" -exec grep "The SAS System" {} \;
find . -name "*.py" -type f -exec grep "something" {} \;
find . -name "*.sh" | xargs grep "ksh"
find /Applications -type d -name "*.app"
find /Path -name "file_name*"
find /Path -name "file_name*" | grep "bar"
find /Path -path "/Path/bar*" -name "file_name*"
find /Path/bar* -name "file_name*"
find /dir -regex '.*2015.*\(album.*\|picture.*\)'
find /dir|egrep '2015.*(album|picture)'
find /etc -atime -1
find /etc -type l -print
find /etc/apache-perl -newer /etc/apache-perl/httpd.conf
find /home/bozo/projects -mtime 1
find /home/pankaj -maxdepth 1 -cmin -5 -type f
find /home/sdt5z/tmp -name "accepted_hits.bam"
find /home/test -name '.ssh'
find /home/user1 -name "*.bin"
find /home/weedly -name myfile -type f -print
find /home/www -type f
find /media/shared \( -iname "*.mp3" -o -iname "*.ogg" \)
find /mnt/raid/upload -mtime -7 -print
find /mnt/raid/upload -mtime +5 -print
find /myfiles -atime +30
find /myfiles -mtime 2
find /myfiles -size 5
find /myfiles -name '*blue*'
find /myfiles -type f -perm -647
find /myfiles -type f -perm -o+rw
find /path -type f -iname "*.txt"
find /path -perm ugo+rwx
find /path ! -perm /020
find /path ! -perm /g+w
find /path -name '*.foo'
find /path ! -perm /022
find /path ! -perm -022
find /path -nouser -or -nogroup
find /path -type f
find /path -perm /ugo+x
find /path/to/dir -name \*.c
find /path/to/directory -type f -mtime 61 -exec rm -f {} \;
find /root -name FindCommandExamples.txt
find /root/ -name myfile -type f
find /root/ -name 'work' -prune -o -name myfile -type f -print
find /root/ -path '/root/work' -prune -o -name myfile -type f -print
find  /storage/sdcard0/tencent/MicroMsg/ -type f  -iname '*.jpg' -print0
find /tmp -user ian
find /tmp -size +10k -size -20k
find /tmp -regex ".*file[0-9]+$"
find /tmp -type f
find /usr -newer /tmp/stamp
find /usr/bin -type f -atime +100
find /usr/bin -type f -mtime -10
find /usr/local/doc -name '*.texi'
find /var/www ! -user apache -print0 | xargs -0
find /var/www -group root -o -nogroup -print0 | xargs -0 chown :apache
find . -name \*.css -print0 | xargs -0 grep -nH foo
find ~/Movies/ -size +1024M
find /res/values-en-rUS /res/xml -iname '*.xml' -print0 | xargs -0 -d '\n' -- grep -i "hovering_msg" --
find images/ -type f
find -L research -maxdepth 2 -type d ! -group ian
find bla -name "*.so"
find . -name file1 -print
find . -name \*.pdf -print
find . -name '*.pdf' -print
find . -name "*.pdf" -print
find . -type f -perm 777 -exec chmod 755 {} \;
find . -name \*.sql
find . -name "*bsd*" -print
find ~/ -maxdepth 3 -name teste.tex
find . -maxdepth 1 -name a\*.html
find . -type f -name "*.php"
find . -maxdepth 1 ! -perm  -o=r
find . -maxdepth 1 -type f -perm -ugo=x
find \( -name 'messages.*' ! -path "*/.svn/*" ! -path "*/CVS/*" \) -exec grep -Iw uint {} +
find \( -name 'messages.*' ! -path "*/.svn/*" \) -exec grep -Iw uint {} +
find -name 'messages.*' -exec grep -Iw uint {} + | grep -Ev '.svn|.git|.anythingElseIwannaIgnore'
find . -type f -print0 | xargs -0 egrep messages. | grep -Ev '.svn|.git|.anythingElseIwannaIgnore'
find . -name 'my*'
find . -name 'my*' -type f
find . -name "*.txt" -exec egrep -l '^string' {} \;
find . -type f -iname *.m4a -print
find . -name "new*.sh"
find . -iname *.mov
find . -iname "*.mov" -follow
find -type d ! -perm -111
find . -type f -exec grep -Iq . {} \; -and -print0 | xargs -0 grep "needle text"
find . -type f -print0 | xargs -0 grep -I "needle text"
find . -type f -print0 | xargs -0 grep -IZl . | xargs -0 grep "needle text"
find . -type f | xargs grep -I "needle text"
find . -type f -exec grep -l 'string' {} \;
find . -mtime 0
find . \( -name en -o -name es \) -prune , -mtime 0 ! -name "*.swp"
find "$(pwd -P)" -mtime 0 -not \( -name '*.swp' -o -regex './es.*' -o -regex './en.*' \)
find -mtime 0 -not \( -name '*.swp' -o -path './es*' -o -path './en*' \)
find . -mtime 0 -not \( -name '*.swp' -o -regex '\./es.*' -o -regex '\./en.*' \)
find . -mtime 0 | grep -v '^\./en' | grep -v '^\./es' | grep -v .swp
find . -mtime 0 | grep -v '^\./en' | grep -v '^\./es'
find . -size +10M -size -50M -print
find . -perm a=rwx,g-w,o-w
find . -perm u=rwx,g=rx,o=rx
find . -perm -o+w
find . type -f -atime 2
find . type -f -amin 2
find . type -f -atime -2
find . type -f -amin -2
find . type -f -atime +2
find . type -f -amin +2
find . type -f -ctime 2
find . type -f -ctime -2
find . type -f -ctime +2
find . type -f -mtime -2
find . type -f -mtime +2
find . type -f -mtime 2
find . -type f -ctime -3 | tail -n 5
find . -type f -name "*.$extension"
find -type f ! -perm -444
find * -type f -print
find -type f -exec grep -Iq . {} \; -and -print
find . -type f -exec grep -Iq . {} \; -and -print
find . -type f -printf '%20s %p\n' | sort -n | cut -b22- | tr '\n' '\000' | xargs -0 ls -laSr
find . -type f \( -iname "*.c" -or -iname "*.asm" \)
find . -type f \( -name "*.c" -o -name "*.sh" \)
find . -type f \( -name "*.conf" -or -name "*.txt" \) -print
find -type f -name "*.wav" | grep export
find . -name "*bash*"
find . -name "*bash*" | xargs
find . -type f \( -name "*cache" -o -name "*xml" -o -name "*html" \)
find . -name "*.VER"
find . -name ".aux"
find . -name '*.log'  -type f  -readable  ! -size 0 -exec sed -n '1{/The SAS System/q0};q1' {} \; -print
find . -name .vendor -prune -o -name '*.rb' -print
find . -name '*.rb' ! -wholename "./vendor/*" -print
find . -name *.php -ctime -14
find . -type f \( -iname "*.txt" ! -iname ".*" \)
find . -type f -name file_name
find . -name '*.java' -newer build.xml -print
find . -name '*.java' -mtime +7 -print
find .  ! -regex ".*[/]\.svn[/]?.*"
find . -not -iwholename '*.svn*'
find . | grep -v \.svn
find . -name "*.rb" -or -name "*.py"
find . -name "*.rb"
find . -regex ".*\\.rb$"
find . -type f -regex ".*\.\(jpg\|jpeg\|gif\|png\|JPG\|JPEG\|GIF\|PNG\)"
find . -name "*.rb" -type f
find ./ -type f -name "*" -not -name "*.o"
find . -type f -iname \*.html -exec grep -s "Web sites" {} \;
find "$PWD" -type d
find $PWD -type d
find . -type d ! -perm -111
find -type d ! -perm -111
find . -type f -executable -print
find -executable
find . -executable -type f
find . -type f -executable -exec file {} \; | grep -wE "executable|shared object|ELF|script|a\.out"
find . -name "a.txt" -print
find -name teste.tex
find . -name '[mM][yY][fF][iI][lL][eE]*'
find . \( -name AAA -o -name BBB \) -print
find . -name test -print
find . -name pro\*
find . ! \( -name "*.exe" -o -name "*.dll" \)
find . -name \*.exe -o -name \*.dll -o -print
find . -not -name "*.exe" -not -name "*.dll"
find . | grep -v '(dll|exe)$'
find .  -perm 775 -print
find . -name "*album*" -a -name "*vacations*" -a -not -name "*2015*"
find . -name "*bash*" | xargs
find . -iname "*needle*"
find . -name "*sh*"
find | egrep string
find . -type f -perm +111 -print
find . -size +10M -size -50M -print
find . -atime +10
find . -regex '.*myfile[0-9][0-9]?'
find . -\( -name "myfile[0-9][0-9]" -o -name "myfile[0-9]" \)
find . -regextype sed -regex '.*myfile[0-9]\{1,2\}'
find . -mtime -5
find . -name "accepted_hits.bam"
find `pwd` -name "accepted_hits.bam"
find -iname 'somename'
find . -name 'Subscription.java'
find . -size -50k
find -type d -exec find {} -maxdepth 1 \! -type d -iname '.note' \;
find . -iname '.note'
find . -iname '.note' | sort
find . -iname '.note' | sort -r
find . ! -name "a.txt" -print
find . -regex ".*/my.*p.$"
find . -regex ".*/my.*p.$" -a -not -regex ".*test.*"
find . -name 'my*'
find ./ -name "*TextForRename*"
find . -name '*bills*' -print
find . -not -name "*.exe" -not -name "*.dll" -not -type d
find . -not -name "*.exe" -not -name "*.dll" -type f
find . -type f ! -name "*1" ! -name "*2" -print
find . -name "*js" -o -name "*rb"
find . -regextype posix-egrep -regex ".*(rb|js)$"
find . -regextype posix-ergep -regex ".*(rb|js)$" -exec grep -l matchNameHere {} \;
find . -iregex ".*packet.*"
find . -name f* -print
find . -not -regex ".*test.*"
find .*
find . -type f \( -iname ".*" ! -iname ".htaccess" \)
find . -type f -name "*.mkv"
find . -type f -ctime -1
find ./ -type f -ls |grep '10 Sep'
find . -type f ! -perm -444
find -type f -ctime +14
find . -mtime -1 ! -name '.DS_Store' -type f -exec basename {} \;
find . -mtime -1 ! -name '.DS_Store' -type f -printf '%f\n'
find ./ -type f -name doc.txt -printf "found\n"
find . -name omit-directory -prune -o -type f
find . -name omit-directory -prune -o -type f  -print
find . \( -name omit-directory -prune -o -type f \) -print
find . \( -name omit-directory -prune \) -o \( -type f -print \)
find -type f -user www
find -type f ! -perm -444
find . -type f -print0 | xargs -0 grep string
find . -type f -printf '"%p"\n' | xargs grep string
find . -type f | xargs grep string
find . -type f -atime $FTIME
find . -type f -ctime $FTIME
find . -type f -mtime $FTIME
find . -name "orapw*" -type f
find -type f -regex ".*/.*\.\(shtml\|css\)"
find . -type f \( -name "*.shtml" -o -name "*.css" \) -print
find . -type f \( -name "*.shtml" -or -name "*.css" \)
find . -type f | egrep '\.(shtml|css)$'
find `pwd` -name "*log" -type f
find . -type f -name "*keep.${SUFFIX}"
find . -type f -name $x
find . -type f -regex ".+-[0-9]+x[0-9]+\.jpg"
find . -type l -name link1
find -P . -lname '*/test*'
find . -lname '*test*'
find . -type l -xtype l
find . -lname '*sysdep.c'
find . -name "*.trc" -ctime +3 -exec ls -l {} \;
find . -name "*.trc" -ctime +3 -exec rm -f {} \;
find . -name "*.trc" -ctime +3 -exec rm {} \;
find .  \( -name work -o -name home \)  -prune -o -name myfile -type f -print
find . -maxdepth 2
find . -name 'foo.cpp' '!' -path '.svn'
find ~/ -name *.png -exec cp {} imagesdir \;
find ~/ -name *.tar.gz -newer filename
find ~/ -mtime -2 -o -newer filename
find ~/ -newer alldata.tar 	-exec tar uvf alldata.tar {} \;
find . -name 'fileA_*' -o -name 'fileB_*'
find ~/dir_data -type f  -exec chmod a-x,u+w {} \;
find $@ -not -name ss
find /path/to/some/dir/*[0-9] -type d -maxdepth 1
find /path/to/directory/folder{?,[1-4]?,50} -name '*.txt'
find $path -type f -name "*.$extension"
find "$d" -mindepth 1 -prune -empty
find /tmp -type f -print0
find $dir -type f
find $root_dir -type f
find / -user olduser  -type f  -exec chown newuser {} \
find / -fstype ext3 -name zsh*
find / -name "*.old" -delete
find / -name "*.old" -exec /bin/rm {} \
find / -name "*~" | grep -v "/media"
find / -size +100M -exec /bin/rm {} \;
find / -fstype ext3 -name zsh -ls
find / -name “*.jpg”
find / – perm -0002
find / -path /proc -prune -o -type f -perm +6000 -ls
find / -path /media -prune -o -size +200000 -print
find / -type f -regextype posix-extended -regex '.*/.{1,24}$'
find / -type f | egrep '.*/.{1,24}$'
find / -type f| egrep -o "/[^/]{0,24}$" | cut -c 2-
find / -type f|egrep "/[^/]{0,24}$"
find / -type f|awk -F'/' '{print $NF}'| awk 'length($0) < 25'
find dirname  -print0 | xargs -0 grep foo
find dirname -exec grep foo {} +
find . -exec grep chrome {} +
find . -exec grep chrome {} \;
find . | xargs grep 'chrome'
find . | xargs grep 'chrome' -ls
find . -exec grep -l foo {} +
find . -exec grep -l foo {} \;
find . -exec grep foo {} +
find . -exec grep foo {} \;
find . -type f -print0 | xargs -0 grep -H 'documentclass'
find . -exec grep -i "vds admin" {} \;
find . -name "string to be searched" -exec grep "text" "{}" \;
find . | xargs grep "searched-string"
find . -name "*bills*" -print0 | xargs -0 grep put
find . -name '*bills*' -exec grep -H "put" {} \;
find . -name '*bills*' -exec grep put {} \;
find /directory/containing/files -type f -exec grep -H 'pattern_to_search' {} +
find /directory/containing/files -type f -print0 | xargs -0 grep "text to search"
history 300 | grep scp | grep important$
find "$directory" -perm "$permissions"
find $HOME -iname '*.ogg' -o -iname '*.mp3'
find ~ -name 'xx*' -and -not -name 'xxx'
find ~/ -atime +10
find ~ -name "test*" -print
find ~/ -name *.tar.gz -newer filename
find ~/ -name '*.txt'
find ~/ -mtime -2 -o newer filename
find $HOME -mtime +365
find ~ -name *.txt
find $HOME -mtime -7
find $HOME -mtime -1
find /home -user sam
find ~ -daystart -type f -mtime 1
find ~ -type f -name '*.mkv' -o -name '*.mp4' -o -name '*.wmv' -o -name '*.flv' -o -name '*.webm' -o -name '*.mov'
find ~ -type f -regex '.*\.\(mkv\|mp4\|wmv\|flv\|webm\|mov\)'
find local /tmp -name mydir -type d -print
find "$absolute_dir_path" -type f -print0
find directory_name -type f -print0 | xargs -0 grep -li word
find folder_name -type f -exec grep your_text  {} \;
find . -type f -exec grep "foo" '{}' \;
find ./ -type f | xargs grep "foo"
find ./ -type f -print -exec grep -n -i "stringYouWannaFind" {} \;
find . -type f -exec grep -n "stuff" {} \; -print
find -type f -exec grep -Hn "texthere" {} +
find . -type f -exec grep -H whatever {} \;
find . -type f | xargs -L 100 grep whatever
find . -type f | xargs grep whatever
find / -group users -iname "filename"
find / -user pat -iname "filename"
find src/ -name '*.[ch]'
find / -xdev -name \*.rpm
find / -type d -name "needle"
find / -group managers -print
find / -user admin -print
find  / -iname findcommandexamples.txt
find  / -name '[a-c]*'
find / -type f -iname "filename"
find . -name myfile |& grep -v 'Permission denied'
find / -name "testfile.txt"
find / -iname "testfile.txt"
find xargstest/ -name 'file??'
find ~ Music -name '*.mp3'
find ~/Books -name Waldo
find ~/Books -type f -name Waldo
find foo bar baz -name "*.rb"
find /usr -name "Chapter*" -type f
find /usr/local -name "*.html" -type f
find /home/user1 -name \*.bin
grep -n 'something' HUGEFILE | head -n 1
apropos postscript | grep -i png
apropos
apropos disk
apropos -s 3 . | grep ^[a-z]
find /usr/local -name "*.html" -type f -exec chmod 644 {} \;
find htdocs cgi-bin -name "*.cgi" -type f -exec chmod 755 {} \;
apropos -s 3 .
find ~/ -name '*.txt' -print0 | xargs -0 wc -w
find -type f -perm -110
find * -maxdepth 0
ping -s www.google.com 2 4
ping -c 4 -q google.comz
ping -c 5 -q 12.34.56.78 | tail -n 2
ping -c 5 -b 10.10.0.255 | grep 'bytes from' | awk '{ print $4 }' | sort | uniq
ping -c 5 -b 10.11.255.255 | sed -n 's/.* \([0-9]\+\.[0-9]\+\.[0-9]\+\.[0-9]\+\).*/\1/p' | sort | uniq
find / -name *.mp3 -fprint nameoffiletoprintto
kill -HUP $(ps -A -ostat,ppid | grep -e '[zZ]'| awk '{ print $2 }')
kill -HUP $( cat /var/run/nginx.pid )
kill -9 16085
kill -9 18581 18582 18583
ps -o uid,pid,cmd|awk '{if($1=="username" && $3=="your_command") print $2}'|xargs kill -15
kill `pstree -p 24901 | sed 's/(/\n(/g' | grep '(' | sed 's/(\(.*\)).*/\1/' | tr "\n" " "`
kill %1
kill $!
kill -s WINCH $$
kill `cat /var/run/DataBaseSynchronizerClient.pid`
ping -c 1 -t 1 192.168.1.1
ping -a 10.100.3.104
ping 8.8.8.8 -I eth9 -c 3 -w 3
fold -1 /home/cscape/Desktop/file  | awk -f x.awk
fold -1 /home/cscape/Desktop/table.sql  | awk '{print $0}'
ping ${ip} -I eth9 -c 1
ping -c 1 $remote_machine
ping -W 1 -c 1 10.0.0.$i | grep 'from' &
ping -c 1 192.168.1.$COUNTER | grep 'ms'
ping -c 1 127.0.0.1 #ping your adress once
ping youhostname.local
ping -w 1 $c
cat my_ips | xargs -i dig -x {} +short
ping -c 2 www.google.com
bg
bg %
find / -links +2 -print
find /usr/share/man/ -regex .*grep*
find /usr/share/man/ -regex grep.*
find .  -mtime +7 -print
find . -type f -print | xargs chmod 444
chmod 644 `find /home/my/special/folder -type f`
find /path -type f -exec chmod 644 {} +;
chmod 644 `find -type f`
chmod 644 `find . -type f`
find . -type f -exec chmod 644 {} \;
find . -type f -print0 | xargs -0 chmod 644
env $(cat .env | xargs) rails
find htdocs -type f -exec chmod 664 {} + -o -type d -exec chmod 775 {} +
chmod 600 file
chmod 644 img/* js/* html/*
find /var/www -type d -print0 | xargs -0 chmod 755
find /var/www -type f -print0 | xargs -0 chmod 644
find foldername -type d -exec chmod 755 {} ";"
find foldername -type f -exec chmod 644 {} ";"
find foldername -exec chmod a+rwx {} ";"
find /opt/lampp/htdocs -type d -exec chmod 711 {} \;
find /opt/lampp/htdocs -type d -exec chmod 755 {} \;
chmod 755 $(find /path/to/base/dir -type d)
find /opt/lampp/htdocs -type f -exec chmod 644 {} \;
sudo chmod 755 $(which node)
find . -type d -exec chmod 2770 {} +
find . -type f -exec chmod 400 {} \;
find . -type d -exec chmod 500 {} \;
find media/ -type f -exec chmod 600 {} \;
find var/ -type f -exec chmod 600 {} \;
find . -type f -perm 755 -exec chmod 644 {} \;
find . -type f -exec chmod 0660 {} +
find media/ -type d -exec chmod 700 {} \;
find var/ -type d -exec chmod 700 {} \;
find . -mindepth 1 -type d -print0 | xargs -0 chmod -R 700
find  . -type d -mindepth 1 -print -exec chmod 755 {}/* \;
find ./default/files -type f -exec chmod ug=rw,o= '{}' \;
find $d -type f -exec chmod ug=rw,o= '{}' \;
find ./default/files -type d -exec chmod ug=rwx,o= '{}' \;
find $d -type d -exec chmod ug=rwx,o= '{}' \;
PS1="`hostname`:\!>"
sudo find foldername -exec chmod a+rwx {} ";"
shopt -s checkwinsize
shopt -s dotglob
shopt -s histverify
set -o pipefail
set -o verbose
set -v
set -o xtrace
set -x
set -x
PROMPT_COMMAND='LAST="`cat /tmp/x`"; exec >/dev/tty; exec > >(tee /tmp/x)'
PS1="`whoami`@`hostname | sed 's/\..*//'`"
DISPLAY=`hostname`:0 skype
env DISPLAY=`hostname`:0 skype
find lib etc debian -name "*.sh" -type f | xargs chmod +x
find arch/x86/usr/sbin arch/x86/usr/X11R6/bin usr/sbin/ -type f | xargs chmod a+x
true
find -gid 1000 -exec chown -h :username {} \;
hostname myServersHostname
hostname $(cat /etc/hostname)
touch -m --date="Wed Jun 12 14:00:00 IDT 2013" filename
find . -type d -exec chmod u=rwx,g=rx,o=x {} \;
find . -type d -name files -exec chmod ug=rwx,o= '{}' \;
find . -name "*rc.conf" -exec chmod o+r '{}' \;
find /git/our_repos -type d -exec chmod g+s {} +
ssh -o ConnectTimeout=3 user@ip
set -e
PS1=`hostname`':\W> '
sudo date --set="Sat May 11 06:00:00 IDT 2013"
touch -r A B
echo "$1"| read -a to_sort
me=`basename -- "$0"`
me=`basename "$0"`
touch -d"$(date --date="@$old_time")" B
touch -d '30 August 2013' *.php
touch -t 200510071138 old_file.dat
PS4='+$(date "+%s:%N") %N:%i> '
PS4='+ $(date "+%s.%N")\011 '
ssh -fNT -L8888:proxyhost:8888 -R22222:localhost:22 officefirewall
ssh $USERNAME@localhost -L 80:localhost:3000 -N
sudo ssh $USERNAME@localhost -L 80:localhost:3000 -N
ssh -R 10022:localhost:22 device@server
ssh -fNT -L4431:www1:443 -L4432:www2:443 colocatedserver
set MAVEN_DEBUG_OPTS=-Xdebug -Xnoagent -Djava.compiler=NONE -Xrunjdwp:transport=dt_socket,server=y,suspend=y,address=8000
architecture="$(uname -m)"
b=`echo "$a" | awk '{ print tolower($1) }'`
b=`echo "$a" | awk '{ print toupper($1) }'`
extract_dir=$(tar -tf $FILE | cut -d/ -f1 | uniq)
filename="`basename "http://pics.sitename.com/images/191211/pic.jpg"`"
finalName=$(basename -- "$(dirname -- "$path")")
fname=`basename $f`
file=$( basename "$1" )
file=`basename "$1"`
path=$(basename $(pwd) | awk '{print tolower($0)}')
path=$(basename $(pwd) | tr 'A-Z' 'a-z' )
path=$(basename $(pwd))
rav=$(echo $var | rev)
source <(echo vara=3)
BZIP2_CMD=`which bzip2`
GZIP="$(which gzip)"
OS=$(uname -s)
OS=`uname -s`
PING=$(ping ADDRESS -c 1 | grep -E -o '[0-9]+ received' | cut -f1 -d' ')
PacketLoss=$(ping "$TestIP" -c 2 | grep -Eo "[0-9]+% packet loss" | grep -Eo "^[0-9]")
value=$(uname -r)
shopt -s extglob
shopt -s extglob
shopt -s globstar
shopt -s dotglob
shopt -s extglob
shopt -s globstar
shopt -s -o nounset
shopt -s nullglob
shopt -s nullglob extglob
shopt -s globstar nullglob
shopt -s globstar nullglob dotglob
ssh -L 1234:remote2:22 -p 45678 user1@remote1
find /mydir \(-mtime +20 -o -atime +40\) -exec ls -l {} \;
find /mydir \(-mtime +20 -o -atime +40\) -exec ls -l {} \;
ls -lrt | tail -n1
find /etc/ -user root -mtime 1
find ~ -perm 777
find $HOME -atime +30
set -ex
awk '{print $1}' file2 | comm -1 -3 file1 - | join file2 -
find /home/pat -iname "*.conf" | less
history | more
ps -ef | grep postgres
myVariable=$(env  | grep VARIABLE_NAME | grep -oe '[^=]*$');
set | grep VARIABLE_NAME | sed 's/^.*=//'
env | uniq | sort -r | grep PATH
date "+%Y-%m-%d"
du -sk $(find . -type d) | sort -n -k 1
find . -type d -exec du -sk {} \; |  sort -n -k 1
find /home -type f -exec file {} \;
find /usr/bin | xargs file
file */*.php | grep UTF
file *.php | grep UTF
find /home -group test
less `find -maxdepth 1 -type f -daystart -mtime -1`
find --version
file /mnt/c/BOOT.INI
w | tr -s " " | cut -d" " -f1,5 | tail -n+3
ls -lt | tr -d 0-9
find . -name something -exec ls -l {} \;
man find
man find
man find
man find
man find
man find
pstree -a -p 20238
uname -a
date -u '+%Y-%m-%dT%k:%M:%S%z'
date -d tomorrow+2days-10minutes
date -ud@0
find -D help
find -maxdepth 1 -not -iname "MyCProgram.c"
find /etc -maxdepth 1 -name "*.conf" | tail
find /etc -maxdepth 2 -name "*.conf" | tail
find / \! -name "*.c" -print
find /mnt/raid -type d -print
find / -size +100M -print
find / -mmin -1 -print
find / -mtime +31 -print
find / \! \( -newer ttt -user wnj \) -print
find / \( -newer ttt -or -user wnj \) -print
find / -newer ttt -user wnj -print
find . -name "*.so" -printf "mv '%h/%f' '%h/lib%f'\n" | less -S
find . -type f \( -name "*.php" -o -name "*.phtml" \) -exec wc -l {} +;
find . -type f -name "*.php" -exec wc -l {} +;
find . -type f | wc -l
find . -maxdepth 1 -type d -print | xargs -I {} echo Directory: {}
find . -maxdepth 1 -type d -print | xargs echo Directories:
echo "$list" | uniq -c
find -version
find /etc/ -user root -mtime 1
who
ifconfig eth0 | grep HWaddr |cut -dH -f2|cut -d\  -f2
pstree user
bunzip2 -c bigFile.bz2 | wc -c
shopt -o extglob
shopt globstar
shopt extglob
shopt compat31
shopt dotglob
shopt nullglob
echo "${line}" | egrep --invert-match '^($|\s*#|\s*[[:alnum:]_]+=)'
awk 'BEGIN{srand();} {printf "%06d %s\n", rand()*1000000, $0;}' | sort -n | cut -c8-
sudo chown -Rf www-data *
read -u 4 -N $char -r -s line
read -rsp $'Press enter to continue...\n'
read -s -p "Enter your password: " passwd
read -rsp $'Press any key or wait 5 seconds to continue...\n' -n 1 -t 5
read -rsp $'Press any key to continue...\n' -n 1 key
read -n1 -s
read -s -N 1 SELECT
read -rsp $'Press escape to continue...\n' -d $'\e'
su -
sort -o $file $file
sort -u -t, -k1,1 file
sort -u -t : -k 1,1 -k 3,3 test.txt
sort -S 50% file
sort -o file1.txt file1.txt
sort -k1,1 -k2,2 -t';' --stable some_data
sort -k1,1 -k2,2 -t';' --stable --unique some_data
sort -u -o file !#$
find -type d -printf '%T+ %p\n' | sort
cd $(find . -maxdepth 1 -type d -name "folder_*" | sort -t_ -k2 -n -r | head -1)
find -mindepth 1 -printf "%T@ %P\n" | sort -n -r | cut -d' ' -f 2- | tail -n +7
diff <(sort $def.out) <(sort $def-new.out)
find / -type f -printf "\n%Ab %p" | head -n 11 | sort -k1M
cat myfile.txt| sort| uniq
finger | sort -u
cat file1 file2 |sort -t. -k 2.1
tac a.csv | sort -u -t, -r -k1,1 |tac
sort file -o !#^
sort foo.txt
rev $filename | sort | uniq -f=N | rev
sort file.txt | rev | uniq -f 10 | rev
diff <(sort file1 -u) <(sort file2 -u)
sort -u FILE
sort -u set1 set2
diff <(sort -n ruby.test) <(sort -n sort.test)
sort
rev   test.txt | sort  -k2 | rev
sort -t$'\t' -k6V -k7n file
sort ips.txt | uniq -c
sort inputfile | uniq | sort -o inputfile
sort temp.txt -o temp.txt
sort temp.txt -otemp.txt
sort $tmp | grep -v ':0'  #... handle as required
source "$( dirname "$( which "$0" )" )/lib/B"
source `which virtualenvwrapper.sh`
source "$( dirname "${BASH_SOURCE[0]}" )/incl.sh"
source $(dirname $0)/incl.sh
split -l ${2:-10000} -d -a 6 "$1"
split -l ${2:-10000} -d -a 6 "$1" "${tdir}/x"
split -l 20 $FILENAME xyz
split -d -l $SPLITLIMT $INFILE x_
split -l $MAX_LINES_PER_CHUNK $ORIGINAL_FILE $CHUNK_FILE_PREFIX
split -l 100 "$SOURCE_FILE"
split -a 5 $file
awk '{if (NR!=1) {print}}' $in_file | split -d -a 5 -l 100000 - $in_file"_"
split --number=l/6 ${fspec} xyzzy.
split -n 1000000 /etc/gconf/schemas/gnome-terminal.schemas
split --lines=50000 /path/to/large/file /path/to/output/file/prefix
split /tmp/files
split -n 10000 /usr/bin/cat
split -n 1000 /usr/bin/firefox
split -n 100000 /usr/bin/gcc
split -l20 ADDRESSS_FILE temp_file_
split -b 500M -d -a 4 INPUT_FILE_NAME input.part.
split --bytes=1500000000 abc.txt abc
split bigfile /lots/of/little/files/here
split --lines $lines_per_file complete.out
split -C 100m -d data.tsv data.tsv.
split -l5000000 data.tsv '_tmp';
split -l 100000 database.sql database-
split -l 100 date.csv
split -a 4 -d -l 50000 domains.xml domains_
split -n l/10 file
split -b 1024m file.tar.gz
split -b 1024m "file.tar.gz" "file.tar.gz.part-"
tail -n +2 file.txt | split -l 4 - split_
tail -n +2 file.txt | split -l 20 - split_
split -b 1M -d  file.txt file
split -b 1M -d  file.txt file --additional-suffix=.txt
split -l 20 file.txt new
split -l 200000 filename
split --suffix-length=5 --lines=1 foo.txt
split -a4 -d -l100000 hugefile.txt part.
split -n2 infile
split -b 10 input.txt /tmp/split-file
split -b 10 input.txt xxx/split-file
split --lines=1 --suffix-length=5 input.txt output.
split -l 100 input_file output_file
split -l 600 list.txt
split -l 200000 mybigfile.txt
split -l5000000 randn20M.csv '_tmp';
split -b 10M -d  system.log system_split.log
split --lines=30000000 --numeric-suffixes --suffix-length=2 t.txt t
split -l9 your_file
split -b 1024m "file.tar.gz" "file.tar.gz.part-"
split -b 1024m file.tar.gz
split -l20 ADDRESSS_FILE temp_file_
find /dev/shm/split/ -type f -exec split -l 1000 {} {} \;
split
split -l 3400000
split --lines=75
cat file1 file2 ... file40000 | split -n r/1445 -d - outputprefix
cat *.txt | tail -n +1001 | split --lines=1000
sed 100q datafile | split -C 1700 -
ls | split -l 500 - outputXYZ.
tar [your params] |split -b 500m - output_prefix
sort --unique emails_*.txt | split --numeric-suffixes --lines=200 --suffix-length=4 --verbose
top
env - scriptname
tmux -2
read -N $BUFSIZE buffer
reads=$(zcat $file.fastq)
a=$( df -H )
set lastdaymonth=`cal $month $year  |tr -s " " "\n"|tail -1`
proc_load_average=$(w | head -1 | cut -d" " -f13 | cut -d"," -f1-2 | tr ',' '.')
proc_load_average=`w | head -1 | cut -d" " -f13 | cut -d"," -f1-2 | tr ',' '.'`
output=$(echo "$output" | tr -d '\' | tr -d '\n')
echo $(basename $(dirname $pathname))
echo $(basename $(dirname $(dirname $pathname)))
awk 'FNR==NR { for(i=2;i<=NF;i++) a[$1][i]=$i; next } { for(j=2;j<=NF;j++) $j-=a[$1][j] }1' File2 File1 | rev | column -t | rev
awk 'FNR==NR { for(i=2;i<=NF;i++) a[$1][i]=$i; next } { for(j=2;j<=NF;j++) $j-=a[$1][j] }1' File2 File1 | rev | column -t | rev
ls | xargs -I {} mv {} {}_SUF
rsync -av /home/user1/ wobgalaxy02:/home/user1/
rsync -rtuv /path/to/dir_b/* /path/to/dir_a
rsync -rtuv /path/to/dir_a/* /path/to/dir_b
rsync -pogtEtvr --progress --bwlimit=2000 xxx-files different-stuff
xargs -I '{}' rm '{}'
xargs -i rm '{}'
cut -d' ' -f1 file.txt | xargs dig +short
NAME=`basename "$FILE"`
NAME=`basename "$FILE" | cut -d'.' -f-1`
awk '{print $1}' file.txt | xargs dig +short
basedir=$(dirname "$(echo "$0" | sed -e 's,\\,/,g')")
mkdir -p `dirname /path/to/copy/file/to/is/very/deep/there` \
cut -d / -f 4- .exportfiles.text | xargs -n 1 dirname
gunzip -t file.tar.gz
find file -chour +1 -exit 0 -o -exit 1
find file -prune -cmin +60 -print | grep -q .
find $LOGDIR -type d -mtime +0 -exec compress -r {} \;
find . | cpio -pdumv /path/to/destination/dirrectory
touch `cat files_to_find.txt`
find . -mmin -15 \( ! -regex ".*/\..*" \)
find . -type f -iname "*.py"
find /root/Maildir/ -mindepth 1 -type f -mtime +14 | xargs rm
ping -D -n -O -i1 -W1 8.8.8.8
set -g mouse on
find .  -maxdepth 1 -type d -iname ".[^.]*" -print0 | xargs -I {} -0 rm -rvf "{}"
find /nas01/backups/home/user/ -type d -name ".*" -print0 -exec ls -lrt {} \;
find /       \( -perm -4000 -fprintf /root/suid.txt '%#m %u %p\n' \) , \              \( -size +100M -fprintf /root/big.txt  '%-10s %p\n' \)
find /       \( -perm -4000 -fprintf /root/suid.txt '%#m %u %p\n' \) , \( -size +100M -fprintf /root/big.txt  '%-10s %p\n' \)
column list-of-entries.txt
set -o nounset
var=`echo $var | awk '{gsub(/^ +| +$/,"")}1'`
find . -type f -maxdepth 1 -not -empty -print0 | xargs -0i cp /dev/null {}
file ~/myfile
ifconfig eth0 up
find /home/ -name 'myfile' -type f | rev | cut -d "/" -f2- | rev | sort -u
crontab -l | sed '/# *\([^ ][^ ]*  *\)\{5\}[^ ]*test\.sh/s/^# *//' | crontab -
gzip -dc archive.tar.gz | tar -xf - -C /destination
gzip -dc data.tar.gz | tar -xvf -
cat www-backup.tar.*|gunzip -c |tar xvf -
find . -maxdepth 1 -type f -name '\.*' | sed -e 's,^\./\.,,' | sort | xargs -iname mv .name name
find . -name '*.gz' -print0 | xargs -0 gunzip
find arch etc lib module usr xpic -type f | xargs chmod -x
shopt -u -o history
shopt -u extglob
gunzip -c bigfile.txt.gz | grep -f patterns.txt | sort | uniq -c
zcat daily_backup.sql.gz| grep -E "'x'|/x/"
zcat file.gz | awk -F'[|"]' '$5>5'
gzip -d --stdout file.gz | bash -s -- "-n wordpress localhost"
zcat file.gz
zcat file.gz | cut -f1 -d, | sort | uniq -c | sort -n
zcat file.gz | grep -o '"searchstring":"[^"]*"'| sort | uniq -c | sort -n
gzip -cd path/to/test/file.gz | awk 'BEGIN{global=1}/my regex/{count+=1;print $0 >"part"global".txt";if (count==1000000){count=0;global+=1}}'
find . -name '*.gz' ! -name '*dvportgroups*' ! -name '*nsanity*' ! -name '*vcsupport*' ! -name '*viclient*' ! -name 'vsantraces*' -exec gunzip -vf {} \;
find . -name "*.gz" -execdir gunzip '{}' \;
ls /homes/ndeklein/mzml/*.gz | xargs -I {} gunzip {}
gunzip test1/*/*.gz
find . -name "*.gz" -exec zcat "{}" + |grep "test"
zcat compressFileName | tar xvf -
zcat input.gz | sed -n 's/.*\(userAgent=[^=]*\) [^ =][^ =]*=.*/\1/p'
zcat input.gz | sed -n 's/.*\(userAgent=[^=]*\) [^ =]\+=.*/\1/p'
zcat input.gz | grep -o 'userAgent=[^=]*' | sed 's/ [^ ]*$//'
zcat small-*.gz | split -d -l2000000 -a 3 - large_
gunzip -c 4.56_release.tar.gz | tar xvf -
zcat file.tar.gz |tar x
gunzip -c myarchive.tar.gz | tar -tvf -
gunzip -c openssl-fips-2.0.1.tar.gz | tar xf ­-
zcat tarball.tar.gz | tar x
gunzip *.gz
find . -name "*.gz" | xargs gunzip
find . -name '*.gz' -exec gunzip '{}' \;
gunzip $empty_variable
find data/ -name filepattern-*2009* -exec tar uf 2009.tar {} ;
find data/ -name filepattern-*2009* -print0 | xargs -0 tar uf 2009.tar
find ~/ -newer alldata.tar -exec tar uvf alldata.tar {} ;
history -w
touch -t `date +%m%d0000` /tmp/$$
touch filename
find . -exec touch {} \;
find . -print -exec touch {} \;
find . -print0 | xargs -0 touch
find /path/to/dir -print0 | xargs -0 touch
find / ! -newer /tmp/timestamp -exec touch {} \;
find /your/dir -type f -exec touch {} +
cat <(yes | tr \\n x | head -c $BYTES) <(sleep $SECONDS) | grep n
yes | tr \\n x | head -c $BYTES | grep n
who --ips /var/log/wtmp | grep '^msw.*127.0.0.1'
ssh -F vagrant-ssh default
grep -b -o $'\x0c' filename | less
echo 'Hello World!' | sed $'s/World/\e[1m&\e[0m/'
gzip -dc input1.vcf.gz input2.vcf.gz | awk 'FNR==NR { array[$1,$2]=$8; next } ($1,$2) in array { print $0 ";" array[$1,$2] }'
awk '{ ... }' <(gzip -dc input1.vcf.gz) <(gzip -dc input2.vcf.gz)
find . -name "*.txt" -exec echo {} \; -exec grep banana {} \;
set -o pipefail
find . -perm 040 -type f -exec ls -l {} \;
ping google.com | awk -f packet_loss.awk
kill -0 $PID
kill $PID
chown -v root "$file"
bzip2 -kv */*/*/*/*/*
bzip2 -kv */*/*/*/*
bzip2 -kv */*
bzip2 -kv */*/*/*/*/*/*/*
bzip2 -kv */*/*/*/*/*/*
bzip2 -kv */*/*
bzip2 -kv */*/*/*
zcat /usr/share/doc/mysql-server-5.0/changelog*.gz | less
gzip --help | less
history | less
history | more
history | vim -
vim <(history)
history | vim -R -
zcat bigfile.z | sed -ne '500,1000 p'
zcat bigfile.z | tail -n +500 | head -501
history | head -n 120 | tail -n 5
man find
man find
man bash | less -p BASH_SOURCE
cat -n file.txt | less
man find
tar -xOf TarFile FileB.gz | zless
find ./ -type f -print0 | xargs -0 -n1 md5sum | sort -k 1,32 | uniq -w 32 -d --all-repeated=separate | sed -e 's/^[0-9a-f]*\ *//;'
fold file | wc -l
cat file.txt | fold
fold -w 80 file.txt
fold -w30 longline
fold -w30 -s longline
fold -w79 yourfile | sed -e :a -e 's/^.\{0,78\}$/& /;ta'
fold -w 10
echo 127.0.0.1 ad.doubleclick.net | sudo tee -a /etc/hosts
echo '2-1' |sudo tee /sys/bus/usb/drivers/usb/unbind
echo '2-1.1.1'|sudo tee /sys/bus/usb/drivers/usb/unbind
echo "Australia/Adelaide" | sudo tee /etc/timezone
echo "Hello, world" | tee /tmp/outfile
echo "Some console and log file message" | tee /dev/fd/3
echo "[some repository]" | sudo tee -a /etc/apt/sources.list
echo -e "\n/usr/local/boost_1_54_0/stage/lib" | sudo tee -a /etc/ld.so.conf
echo 'deb blah ... blah' | sudo tee --append /etc/apt/sources.list
sudo echo "deb http://downloads-distro.mongodb.org/repo/ubuntu-upstart dist 10gen" | sudo tee -a /etc/apt/sources.list.d/10gen.list
echo "error" | tee
echo "fifo forever" | cat - fifo | tee fifo
echo foo | readlink /proc/self/fd/1
echo foo | readlink /proc/self/fd/0
echo "hello world" | tee >(wc)
echo suspend | sudo tee /sys/bus/usb/devices/usb3/power/level
echo "myname=\"Test\"" | sudo tee --append $CONFIG
seq 1 10 | sort -R | tee /tmp/lst |cat <(cat /tmp/lst) <(echo '-------') **...**
sudo cat /sys/kernel/debug/tracing/trace_pipe | tee tracelog.txt
ls |& tee files.txt
cat infile | paste -sd ',\n'
cat infile | paste -sd '  \n'
find .
find . -print
bash myscript.sh 2>&1 | tee output.log
ls -a | tee output.file
ls -hal /root/ | sudo tee /root/test.out
ls -lR / | tee -a output.file
ls -lR / | tee output.file
tee /tmp/arjhaiX4
tee foobar.txt
comm -12 <(cut -d " " -f 3 file1.sorted | uniq) <(cut -d " " -f 3 file2.sorted | uniq) > common_values.field
echo $(date) "0" | tee -a log.csv
echo $(date) "1" | tee -a log.csv
tail -F xxxx | tee -a yyyy &
false | tee /dev/null
set -v
set -x
find . -name "*rc.conf" -exec chmod o+r '{}' \;
find . -perm 600 -print | xargs chmod 666
find . -mtime -7 \( '*.jpg' -o -name '*.png' \)
find . -name '*.mp3' -name '*.jpg' -print
tar czvf mytarfile.tgz `find . -mtime -30`
find . -mtime -1 -type f -exec tar rvf "$archive.tar" '{}' \;
sudo find / -name file.txt
find /mydir -atime +100 -ok rm {} \;
tac error.log | awk '{if(/2012/)print;else exit}'
bind '"\C-i":complete'
bind $'"\x61"':self-insert
bind '"\e[24~":"foobar"'
bind '"\e[24~":"pwd\n"'
find ~/tmp -mtime 0 -exec du -ks {} \; | cut -f1
find . -name "*jpg" -exec du -k {} \; | awk '{ total += $1 } END { print total/1024 " Mb total" }'
find htdocs cgi-bin -name "*.cgi" -type f -exec chmod 755 {} \;
find htdocs cgi-bin -name "*.cgi" -type f -exec chmod 755 {} \;
chown -R :daemon /tmp/php_session
chown -R :lighttpd /var/lib/php/session
chown :friends myfile
find /usr/local -name "*.html" -type f -exec chmod 644 {} \;
find . -maxdepth 1 -not -name "." -print0 | xargs --null chown -R apache:apache
find ~ -group vboxusers -exec chown kent:kent {} \;
chown -R andrewr:andrewr *
chown -v root:root /path/to/yourapp
chown user:group file ...
chown root:specialusers dir1
chown root:root it
chown root:root script.sh
sudo chown root:root uid_demo
find . -not -iwholename './var/foo*' -exec chown www-data '{}' \;
chown user_name file
sudo chown root /home/bob
chown user destination_dir
$sudo chown root file.sh
sudo chown el my_test_expect.exp     //make el the owner.
sudo chown root process
find /mydir -type f -name "*.txt" -execdir chown root {} ';'
find /mydir -type f -name "*.txt" -print0 | xargs -0 chown root $(mktemp)
find dir_to_start -name dir_to_exclude -prune -o -print0 | xargs -0 chown owner
find / -type f -perm 0777 -print -exec chmod 644 {} \;
find / -type f -perm 0777 -print -exec chmod 644 {} \;
chmod 751 `find ./ -type d -print`
find /home/john/script -name "*.sh" -type f -exec chmod 644 {} \;
find /path/to/directory -type f -mtime +30 -exec chmod 644 {} +
cd `find a |sed '$!d'`
find /the/path -depth -name "*.abc" -exec rename 's/\.abc$/.edefg/' {} +
find /the/path -type f -name '*.abc' -execdir rename 's/\.\/(.+)\.abc$/version1_$1.abc/' {} \;
find -name ‘*.lst’ -exec rename .lst a.lst {} \;
find . -xtype f \! -iname *.html   -exec mv -iv "{}"  "{}.html"  \;
find . -type d -exec chgrp usergroup {} \;
find . -type f -exec chgrp usergroup {} \;
find / -group 999 -exec chgrp NEWGROUP {} \;
find /u/netinst -print | xargs chgrp staff
find / -user edwarda -exec chgrp pubs "{}" \;
find . -name "*" -exec chgrp -v new_group '{}' \; -exec chmod -v 770 '{}' \;
find . -name "*" \( -exec chgrp -v new_group {} \; -o -exec chmod -v 770 {} \; \)
find . /home/admin/data/ -type d -exec chown admin.admin {} \;
find /usr/lpp/FINANCIALS -print | xargs chown roger.staff
find . /home/admin/data/ -type f -exec chown admin.admin {} \;
find /u/netinst -print | xargs chown netinst
find . -exec chown myuser:a-common-group-name {} +
find / -user 999 -exec chown NEWUSER {} \;
find .-type f -user root -exec chown tom {} \;
find -gid 1000 -exec chown -h :username {} \;
find . -type d -exec chown username {} \;
find . -type f -exec chown username {} \;
find . -type f | xargs chown username
find . -type f -ok chown username {} \;
find . -type f -print0 | xargs -0 chown username
find / -user edwarda -exec chown earnestc "{}" \;
find / -user edwarda -print | xargs chown earnestc
find . -type d -exec chmod 755 {} \;
find . -name "*.php" -exec chmod 755 {} \;
find . -name "*.php" -exec chmod 755 {} +
find -type d -exec chmod 755 {} \;
find . \( -type f -exec sudo chmod 664 "{}" \; \) , \( -type d -exec sudo chmod 775 "{}" \; \)
find /home/user/demo -type f -perm 777 -print -exec chmod 755 {} \;
find -type f -exec chmod 644 {} \;
find /var/www/ -type f -iname "*.php" -exec chmod 700 {} \;
find /home -type f -perm 0777 -print -exec chmod 700 {} \;
find . -type f -perm 777 -exec chmod 755 {} \;
find / -name *.rpm -exec chmod 755 '{}' \;
sudo find . -type d -exec chmod 755 {} +
find . -type f -exec chmod 664 {} \;
find . -type f | xargs chmod 664
find . -type f -print0 | xargs -0 chmod 664
chmod 640 `find ./ -type f -print`
find . -type d -exec chmod 775 {} \;
chmod 751 `find ./ -type d -print`
find . -type d -exec chmod 2775 {} \;
find . -type d | xargs chmod 2775
find . -type d -print0 | xargs -0 chmod 2775
sudo find /path/to/someDirectory -type d -print0 | xargs -0 sudo chmod 755
find root_dir -type d -exec chmod 555 {} \;
find /home/nobody/public_html -type d -exec chmod 755 {} \;
find /store/01 -name "*.fits" -exec chmod -x+r {} \; \
find /store/01 -name "*.fits" -exec chmod -x+r {} \; -exec ls -l {} \; | tee ALL_FILES.LOG
find /path/to/dir/ -type f -print0 | xargs -0 chmod 644
find . -type f -exec chmod 500 {} ';'
find root_dir -type f -exec chmod 444 {} \;
chmod 640 `find ./ -type f -print`
sudo find . -type f -exec chmod 644 {} +
sudo find /path/to/someDirectory -type f -print0 | xargs -0 sudo chmod 644
find /home/nobody/public_html -type f -exec chmod 644 {} \;
find . -type f -exec chmod 664 {} \;
find /var/ftp/mp3 -name '*.mp3' -type f -exec chmod 644 {} \;
find /var/www/html -type d -perm 777 -print -exec chmod 755 {} \;
find . -name '*-GHBAG-*' -exec rename 's/GHBAG/stream-agg/' {} +
sudo chown root:dockerroot /var/run/docker.sock
chown amzadm.root  /usr/bin/aws
sudo chown root:wheel bin
chown owner:nobody public_html
find --version
find myfile -perm 0644 -print
find . -path ./.git -prune -o -print -a \( -type f -o -type l -o -type d \) | grep '.git'
bind -p | grep $'"\x61"'
find /usr/bin | xargs file
find . -name '*.h' -execdir diff -u '{}' /tmp/master ';'
find . \! -name "*.Z" -exec compress -f {} \;
find . -depth -print | cpio -dump /backup
head -n99999999 file1.txt file2.txt file3.txt
date -d @1278999698 +'%Y-%m-%d %H:%M:%S'
find . -type f -name "*.mp3" -exec cp {} /tmp/MusicFiles \;
find /raid -type d -name ".local_sd_customize" -ok cp /raid/04d/MCAD-apps/I_Custom/SD_custom/site_sd_customize/user_filer_project_dirs {} \;
cp `find -perm -111 -type f` /usr/local/bin
find "$sourcedir" -type f -name "*.type" | xargs cp -t targetdir
find . -type f -mtime +30 -name "*.log" -exec cp {} old \;
find . -name '*.mp3' -exec cp -a {} /path/to/copy/stuff/to \;
find . -type f -exec cp {} /tmp +
find /tmp -type f -mtime -30 -exec cp {} /tmp/backup \;
find . | cpio -pdumv /path/to/destination/dir
find dir1 dir2 dir3 dir4 -type d -exec cp header.shtml {} \;
find dir1 dir2 dir3 dir4 -type d -exec cp header.shtml {} \;
find /usr/src -name "*.html" -exec grep -l foo '{}' ';' | wc -l
find . -type f | wc -l
jobs | wc -l
find . \( -name "*.c" -or -name "*.cpp" -or -name "*.h" -or -name "*.m" \) -print0 | xargs -0 wc
find . \( -name "*.c" -or -name "*.cpp" -or -name "*.h" -or -name "*.m" -or -name '*.java' \) -print0 | xargs -0 wc
find . -name "*.java" -print0 | xargs -0 wc
find . -type f -exec wc -l {} \; | awk '{total += $1} END{print total}'
find . -name .snapshot -prune -o \( \! -name *~ -print0 \) | cpio -pmd0 /dest-dir
find . -cpio /dev/fd0 -print | tee /tmp/BACKUP.LOG
find /home -depth -print | cpio -ov -0 /dev/rmt0 | tee -a tape.log
find source/directory -ctime -2 | cpio -pvdm /my/dest/directory
tar -zcvf compressFileName.tar.gz folderToCompress
tar -cvzf filename.tar.gz folder
tar -C my_dir -zcvf my_dir.tar.gz .[^.]* ..?* *
tar -N '2014-02-01 18:00:00' -jcvf archive.tar.bz2 files
find . -depth -print | cpio -o -O /target/directory
gzip `find . \! -name '*.gz' -print`
find . \! -name "*.gz" -exec gzip {} \;
sudo  ln  -d  existing_dir  new_hard_link
find \( -name "*.htm" -o -name "*.html" \) -a -ctime -30 -exec ln {} /var/www/obsolete \;
find . -type f -exec md5 {} \;
find -iname "MyCProgram.c" -exec md5sum {} \;
find /media/Movies -type f -mtime -30 -exec ln -s {} /media/Movies/New/ \;
ln -s "../config/environments"
ln /media/public/xampp/mysql/data/my_db -s
sudo ln -s /usr/include/oracle/11.2/client $ORACLE_HOME/include
ln -s -- ./local--pdf-kundendienst -pdf-kundendienst
ln -s /usr/share/my-ditor/my-editor-executable /usr/bin/my-editor
sudo ln -s /usr/lib/jvm/java-7-oracle /usr/lib/jvm/default-java
ln -s .bashrc test
ln -s www1 www
ln -s "/cygdrive/c/Program Files" /cygdrive/c/ProgramFiles
ln -sf '/cygdrive/c/Users/Mic/Desktop/PENDING - Pics/' /cygdrive/c/Users/Mic/mypics
find . -name *.pdf | xargs tar czvf /root/Desktop/evidence/pdf.tar
find . \( -iname "*.png" -o -iname "*.jpg" \) -print -exec tar -rf images.tar {} \;
find . -name -type f '*.mp3' -mtime -180 -print0 | xargs -0 tar rvf music.tar
find . \( -iname "*.png" -o -iname "*.jpg" \) -print -exec tar -rf images.tar {} \;
find / -name *.jpg -type f -print | xargs tar -cvzf images.tar.gz
find . -maxdepth 2 -size +100000 -exec bzip2 {} \;
find . -name '*.log' -mtime +3 -print0 | xargs -0 -P 4 bzip2
find . -name '*.log' -mtime +3 -print0 | xargs -0 -n 500 -P 4 bzip2
tar -czf backup.tar.gz -X /path/to/exclude.txt /path/to/backup
tar -I 7zhelper.sh -cf OUTPUT_FILE.tar.7z paths_to_archive
tar -I pbzip2 -cf OUTPUT_FILE.tar.bz2 /DIR_TO_ZIP/
tar -I pbzip2 -cf OUTPUT_FILE.tar.bz2 paths_to_archive
tar cf - $PWD|tar tvf -
tar cf - $PWD|tar tvf -|awk '{print $6}'|grep -v "/$"
tar czfP backup.tar.gz /path/to/catalog
find -name "*.txt" cp {} {}.bkup \;
mkdir a b c d e
mkdir bravo_dir alpha_dir
mkdir foo bar
mkdir mnt point
mkdir .hiddendir
mkdir /cpuset
sudo mkdir /data/db
mkdir /etc/cron.15sec
mkdir /etc/cron.5minute
mkdir /etc/cron.minute
mkdir /path/to/destination
mkdir /tmp/foo
mkdir /tmp/new
sudo mkdir /var/svn
mkdir TestProject
mkdir aaa
mkdir aaa/bbb
mkdir backup
mkdir certs/
mkdir destdir
mkdir -p dir
mkdir dir1
mkdir -m 777 dirname
mkdir -p es/LC_MESSAGES
mkdir -p foo
mkdir foo
mkdir ~/log
mkdir new_dir
mkdir ~/practice
mkdir ~/public_html
mkdir saxon_docs
mkdir subdirectory
mkdir tata
mkdir ~/temp
mkdir testExpress
find /original -name '*.processme' -exec echo ln -s '{}' . \;
ln -s $(echo /original/*.processme) .
find /incoming -mtime -5 -user nr -exec ln -s '{}' /usr/local/symlinks ';'
find /your/source/dir/ -iname '*.txt.mrg' -exec ln -s '{}' /your/dest/dir/ \;
find / -name *.jpg -type f -print | xargs tar -cvzf images.tar.gz
ln -sf "$(readlink -f "$link")" "$link"
find . -empty -exec rm '{}' \;
find "$DIR" -type f -atime +5 -exec rm {} \;
find ~/ -name 'core*' -exec rm {} \;
find . -name "*.bam" | xargs rm
find . -name bad -empty -delete
find . -type f -empty -delete
find . -type f ! -iname "*.txt" -delete
find / -type f -name "*.txt" -print | xargs rm
find $HOME/. -name "*.txt" -ok rm {} \;
find kat -type f \( -name "*~" -p -name "*.bak" \) -delete
find . -name '*.doc' -exec rm "{}" \;
find . \( -name '*.wmv' -o -name '*.wma' \) -exec rm {} \;
find . -name "*.bak" -delete
find -L /usr/ports/packages -type l -exec rm -- {} +
find /prog -type f -size +1000 -print -name core -exec rm {} \;
find . -type d -empty -delete
find . -type d -empty -exec rmdir {} \;
find . -empty -exec rm {}\;
find . -empty -ok rm {}\;
find . -empty -delete -print
find . -depth -type d -empty -exec rmdir {} \;
find . -maxdepth 1 -type d -empty -exec rm {} \;
find -name '*~' -delete
find -name '*~' -print0 | xargs -0 rm
find . -name "*~" -print | xargs rm
find . -delete
find . -print0 | xargs -0 rm
find / -nouser -exec rm {}\;
find . -size +1024 ?print|xargs -i rm \;
find . -nouser | xargs rm
find . ( -name '*.bak' -o -name *.backup ) -type f -atime +30 -exec rm '{}' ;
find . -mtime -14 -print|xargs -i rm \;
find / -user edwarda -exec rm "{}" \;
find / -user edwarda -ok rm "{}" \;
find . -type f -name "Tes*" -exec rm {} \;
find -name '*.log' -delete
find ./ -name '*.log' -print0 | xargs -0 rm
find ./ -name '*.log' | xargs rm
find . — name "*.LOG" — mtime +5 -ok rm {} \;
find . -type f -name "*.mp3" -exec rm -f {} \;
find /home/ -exec grep -l “mp3” {} \; | xargs rm
find /home -type f -name *.mp4 -size +10M -exec rm {} \;
find / -type f -print0 | xargs -0 grep -liwZ GUI | xargs -0 rm -f
find . -maxdepth 1 -type f -delete
find . -type f -print -delete
find /var/www/*.php -type f -exec rm {} \;
find /tmp/ -ctime +15 -type f -exec rm {} \;
find /tmp/ -type f -mtime +1 -delete
find /tmp/ -type f -mtime +1 -exec rm {} \;
find /tmp/ -type f -mtime +1 -print0 | xargs -0 -n1 rm
find /tmp/ -type f -mtime +1 -exec rm {} +
find . -name "*.txt" -ok rm {} \;
find . -type f -name "*.txt" -delete
find . -type f -name "*.txt" -exec rm -f {} \;
find / -name "oldStuff*.txt" -delete
find /tmp -name "*.tmp" | xargs rm
find /tmp -name "*.tmp" -print0 | xargs -0 rm   find /tmp -name "*.tmp" -print0 | xargs -0 rm
find $DBA/$ORACLE_SID/bdump/*.trc -mtime +7 -exec rm {} \;
find /dirpath \( -name \*.trc -a -mtime +30 \) -exec rm {} \;
find . -iname .svn -exec rm -rf {} \;
find . -iname .svn -print | xargs rm -rf
find . -iname .svn -print0 | xargs -0 rm -rf
bind '"\e[24~":"\C-k \C-upwd\n"'
find . -size +100000 -ls
find . -mtime -14 -ls
find . -type f -empty
find $@ -ls
find . \( -name '*jsp' -o -name '*java' \) -type f -ls
find . -type f -ls
find ~ -type f -mtime 0 -ls
find / -nogroup \( -fstype jfs -o -fstype jfs2 \) -ls
find / -nouser \( -fstype jfs -o -fstype jfs2 \) -ls
find / -path /proc -prune -o -type f -perm +6000 -ls
find / -size +1000 -mtime +30  -exec ls -l {} \;
find / -type f -user root -perm -4000 -exec ls -l {} \;
find "$STORAGEFOLDER" -name .todo -printf '%h\n' | uniq | xargs ls -l
find "$STORAGEFOLDER" -name .todo -printf '%h\n' | xargs ls -l
find /mydir \(-mtime +20 -o -atime +40\) -exec ls -l {} \;
find . -size +10k -exec ls -l {} \;
find ~ -iname '*.jpg' -exec ls {} \;
find ~ -iname '*.jpg' -exec ls {} +
find -exec grep -q fuddel {} ";" -exec grep -q fiddel {} ";" -ls
find / -size +1000k -exec ls -l {} \; -print
find . -mmin -60 -ls
find . -iname "Articles.jpg" -exec ls -l {} \;
find . -iname "Articles.jpg" -print0 | xargs -0 ls -l
find . -mmin -60 -type f -exec ls -l {} \;
find . -mmin -60 -type f -ls
find . -mmin -60 -type f | xargs ls -l
find /home -name Trash -exec ls -al {} \;
find . -type d -ls
find . -type d -exec ls -algd {} \;
find / -print0 -type d | xargs -0 ls -al
find / -type f -size 0 -exec ls -l {} \;
find /var -size +10000k -print0 | xargs -0 ls -lSh
find . -name  * -exec ls -a {} \;
find . — type f -exec ls -1 {} \;
find . -size +10k -exec ls -ls {} \+ | sort -nr
find -daystart   -atime 0 -ls
find . -size +10k -exec ls -lh {} \+
find . -mtime -1 -ls
find . -mtime -1 | xargs ls -ld
find /home -size +200M -exec ls -sh {} \;
find / -name 'Metallica*' -exec ls -l {} \;
find . -type f -name '*.java' -ls | sort -k +7 -r
find . -size +1000k -name *.log -print0 | xargs -0 ls –lSh
find . -type f -print0 | xargs -0 ls -l
find . -maxdepth 1 -type f -exec ls -l {} \; | less
find . -type f -ls
find / -type f \( -perm -4000 -o -perm -2000 \) -exec ls -l {} \;
find / -regex ".*\.\(xls\|csv\)"
find / -type f \( -name "*.xls" -o -name "*.csv" \) -exec ls -l {} \;
find . -mmin 60 -print0 | xargs -0r ls -l
find . -mmin -60 -type f -exec ls -l {} +
find . -mmin -60 |xargs ls -l
find /etc -type d -print
find -type d
find . -type d -print0
find . -maxdepth 1 -mindepth 1 -type d
find . -iregex '.*/.git/.*' -prune -o -type d -name 'CVS'
find -type d
find / -type d -print
find "$ORIG_DIR" -name "*" -type d
find /myfiles -type d
find /PROD -maxdepth 1 -type d
find Symfony -type d
find .vim/ -maxdepth 1 -type d
find -type d -and -atime +3
find . -size 0k
find ~ -empty
find /home -perm /a=x
find /dir/to/search/ -not -name "*.c" -print
find /dir/to/search/ \! -name "*.c" print
find $HOME -not -iname "*.c" -print
find $HOME \! -iname "*.c" print
find . -printf '%p '
find . ! — type d -print
find . -name \*.ext | cat - list.txt | sort | uniq -u
find . -type f -name '*.ini'
find "/proc/$pid/fd"
find .
find -L .
find -regex "^.*~$\|^.*#$"
find . \! -name '.'
find . ! -name "*.txt"
find . ! -name '*git*' | grep git
find folder1/ -type f -printf "%d\t%p\n" | sort -n | sed -e "s|[0-9]*\t||"
find -regex "$rx"
find . -size +1M
find . -size +100k -a -size -500k
find -mmin 60
find -mmin +60
find . -print0
find * -maxdepth 0 -name "efence*" -prune -o -print
find .
find . -print
find . -iname "*$@*" -or -iname ".*$@*"
find . -prune -print
find . -printf "%h/%f : dernier accès le %Ac\n"
find . -printf "%h/%f : dernier accès le %AA %Ad %AB %AY à %AH:%AM:%AS\n"
find . -maxdepth 0
find . -name "*.txt" -prune -o -print
find . -name \? -mtime -1
find . ! -size 0k
find . ! -user john
find . -not -regex ".*test.*"
find . -\( -name "myfile[0-9][0-9]" -o -name "myfile[0-9]" \)
find . -regextype sed -regex '.*myfile[0-9]\{1,2\}'
find . -regex '.*myfile[0-9][0-9]?'
find -name met*
find . -name test -prune -o -print
find . -name test -prune
find /dir -amin -60
find /dir -cmin -60
find /
find / -noleaf -wholename '/proc' -prune -o -wholename '/sys' -prune -o -wholename '/dev' -prune -o -perm -2 ! -type l  ! -type s ! \( -type d -perm -1000 \) -print
find / -name /transfer -prune -o -print
find / -size +50M -iname "filename"
find /usr -maxdepth 1 -print
find /usr/src ! \( -name '*,v' -o -name '.*,v' \) '{}' \; -print
find bar -path /foo/bar/myfile -print
find . -type f -newermt "2014-01-01" ! -newermt "2014-06-01"
find . -type f -name ".*"
find -name "*.htm" -print
find /usr -name tkConfig.sh
find . -name 'foo.cpp' '!' -path '.svn'
find / -name .profile -print
find -iname "*.jpg"
find / -name “*.mp3” -atime +01 -type f
find . \! -empty -type d
find . -type f | tac
find /home/the_peasant -type f
find teste1 -type f
find . -type f
find . -type f
find . -type f -print0
find . -type f
find . -type f -print0
find . -type f print0 | sort -r
find . -type f -readable
find "$ORIG_DIR" -name "*" -type f
find pathfolder -type f
find "$ORIG_DIR" -name "*" -type d -o -name "*" -type f
find . -name *.pdf
find . -name '*.php' -o -name '*.xml' -o -name '*.phtml'
find . \( -type d -regex '^.*/\.\(git\|svn\)$' -prune -false \) -o -type f -print0
find -type f -name "* *"
find image-folder/ -type f
find $directory -type f
find . -type f
find tmp -type f -printf "f %s %p\n"
find -type f -name *ummy
find . -type f -atime 7
find . -type f -atime -7
find . -type f -atime +7
find -type f -name dummy
find dir -type f -printf "f %s %p\n"
find /Users/david/Desktop/-type f
find Symfony -type f
find -type f -and -mmin -30
find . -path "*src/main*" -type f -iname "*\.scala*"
find . -type f -path "*src/main/*\.scala"
find . -type f -regex ".*src/main.*\.scala$"
find . -name "*.sh"
find . -type l
find ./ -name "*.sqlite" -printf '%Tc %p\n'
find . -lname "*"
find /myfiles -type l
find -L /myfiles
find $target -type f -iname "*.txt"
find . -name "*.txt"
find . -name ".txt"
find . -type f -name "*.txt"
find /home/you -iname "*.txt" -mtime -60 -print
find "/tmp/1" -iname "*.txt"
find /tmp/1 -iname '*.txt' -not -iname '[0-9A-Za-z]*.txt'
find /user/directory/* -name "*txt" -mtime 0 -type f
find /Users/david/Desktop -type f \( -name '*.txt' -o -name '*.mpg' -o -name '*.jpg' \)
find . -name \*.c -print
find . \( ! -name . -prune \) -name "*.c" -print
find .  -name .svn -prune -o -name "*.c" -print
find /home/david -atime -2 -name '*.c'
find /home/david -amin -10 -name '*.c'
find . -name "*.mov"
find . -iname "*.mov" -printf "%p %f\n"
find -name *.sh
find . -name "*.c"
find . -type f \( -iname "*.sh" -or -iname "*.pl" \)
find . -type f \( -name "*.[sS][hH]" -o -name "*.[pP][lL]" \)
find /usr -name '*.sh'
find /usr -name \*.sh
find euler/ -iname "*.c*" -exec echo {} \; -or -iname "*.py" -exec echo {} \;
find kat -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.h" \)
find /some/dir -maxdepth 1 \( -name '*.c' -o -name '*.h' \) -print
find /etc -name "*.conf" -printf "%f accessed %AF %Ar, modified %TF %Tr\n"
find .  -path '*/*config'
find /etc -name '*.conf'
find . -size 0 -printf '%M %n %u %g %s %Tb\n \b%Td %Tk:%TM %p\n'
find . -type d
find . -type d -print
find -type d|sed -r '/^.$/{s:.:#!/bin/bash:};{s/^\./mkdir -p &/}'
find -type d -path '.svn' -prune -o -print
find . -type d -name aa -prune
find . -type d -name aa -prune -o -print
find . -name secret -type d -prune -o -print
find . ! -name "node_modules" -type d
find . -type d -atime +2
find . -mindepth 1 -type d -print0
find . -type d -regextype posix-egrep -regex '\./processor[0-9]*/10\.(1|2)'
find . -type d -regextype posix-egrep -regex '\./processor[[:digit:]]*/10\.(1|2)'
find /path/to/dest -type d \( ! -name tmp \) -o \( ! -name cache \) -print
find /path/to/dest -type d \( ! -name tmp \) -print
find /usr/share -type d
find /var -maxdepth 2 -type d;
find master -type d | sort
find . -name '*.doc'
find . -empty
find . -empty
find / -size 0 -print
find /opt -type f -empty
find /home/david -name 'index*'
find /home/david -iname 'index*'
find . -maxdepth 1 -type d \( ! -name . \)
find . -not -name "*.pl"
find  .  -path  './src/emacs'  -prune  -o -print
find . -name "*:*"
find /home ! -group test -printf "%p:%g\n"
find /usr/src -name CVS -prune -o -depth +6 -print
find . -name .snapshot -prune -o -name '*.foo' -print
find . \( -name .snapshot -prune -o -name '*.foo' \) -print
find /var/adm/logs/morelogs/* -type f -prune -name "*.user" -print
find /var/adm/logs/morelogs/* -type f -prune \( -name "admin.*" -o -name "*.user" -o -name "*.user.gz" \) -print
find . -name \*.c -print0
find . -name "filename including space"
find . -name "filename including space" -print0
find . -type f -name "*searched phrasse*" ! -path "./tmp/*" ! -path "./var/log/*"
find . | sed 's/.*/& &/'
find .
find . -size +10k
find . -atime -15
find . -cmin +2 -cmin -6
find . -ctime -1 -print
find . -mtime +7
find . -perm /222
find . -name "file2015-0*"
find /usr -perm 0777 -print
find . -maxdepth 1 -print0
find . -maxdepth 0 -print
find . \( ! -path "*target*" -a ! -path "*tools*" -a ! -path "*.git*" -print \)
find . -name 'secret' -prune -o -print
find .
find . -print
find . | awk '{ print "FILE:" $0 }'
find . | xargs echo
find ./
find . -type f -printf "%C@ %p\n" | sort -rn | head -n 10
find . — name "*" — print -о -name ".*" — print -depth
find . -type f -printf '%T@ %p\n' | sort -n | tail -10 | cut -f2- -d" "
find . -maxdepth 1 -type f | xargs -I ‘{}’ sudo mv {} /directory1/directory2
find . -type d -depth
find . -name PERSONAL -prune -o -print
find . ! -path *mmm*
find . -type d ! -name aa
find . \! -path "./.git*" -a \! -name states_to_csv.pl
find . -name mmm -prune -o -print
find /target/ | grep -v '\.disabled$' | sort
find . \( -name 'secret' -a -prune \) -o -print
find . ! -wholename "./etc*"
find . \( -type d -name aa -prune \) -o \( -type f -name 'file*' -print \)
find . ! -path  "*.git*" -type f -print
find . -not -name "*.pl" -not -name "*.sh" -not -name "*.py"
find . -type f -atime -1
find . -type f -atime +1
sudo find . -print0
find . -mtime +30 -a -mtime -7 -print0
find . -name "*.ksh" -prune
find  -mtime -1
find \( -size +100M -fprintf /root/big.txt %-10s %p\n \)
find . -path './sr*sc'
find .  -path '*f'
find . -path './sr*sc'
find . -path './src/emacs' -prune -o -print
find . | xargs grep -PL "\x00" | xargs grep -Pl "\x0c"
find . -nogroup
find . -nouser
find . -name "*.bash"
find . -path './kt[0-9] '
find . -size +1024 -print
find . -amin -60
find . -newer /bin/sh
find . -newermt “Sep 1 2006” -and \! -newermt “Sep 10 2006”
find .  -newermt "1 hour ago"
find . -mtime -14 -print
find . -mtime -2
find . -mtime -1
find . -mtime -1 -print
find . \( -type d ! -name . -prune \) -o \( -mtime -1 -print \)
find . -mtime -5
find -mtime +7 -print | grep -Fxvf file.lst
find . -perm 777  -mtime 0 -print
find . -perm 777 -a -mtime 0 -a -print
find . -name '*bills*' -print
find . -name 'fileA_*' -o -name 'fileB_*'
find .
find . -name modules
find / -name "*" — print
find / -type f -exec echo {} \;
find / -size +10000k
find / -name "apache-tomcat*"
find / -perm -u+s -print
find / \! -name "*.c" -print
find / -newerct '1 minute ago' -print
find / -fstype nfs -print
find / -size 20
find / -nogroup staff -print
find / -nouser -print
find / -group lighttpd -print
find / -user user1
find / -newer ttt -user wnj -print
find / \( -newer ttt -or -user wnj \) -print
find / -uid 1005
find / \! \( -newer ttt -user wnj \) -print
find / -mmin -10
find "$ORIG_DIR"
find /Users/Me/Desktop -user popo -perm 777
find /Users/Me/Desktop -readable
find /dev -user "peter" |more
find /home/mywebsite -type f -ctime -7
find /etc /srv \! -path "./srv/tftp/pxelinux.cfg*" -a \! -name /etc/mtab
find /home ! -group test
find /home -not -group test
find /home -perm /u=s
find /mp3-collection -name 'Metallica*' -or -size +10000k
find a
find /etc /srv \( -path /srv/tftp/pxelinux.cfg -o -path /etc/mtab \)  -prune -o -print
find /mydir1 /mydir2 -size +2000 -atime +30 -print
find $HOME -print
find /home ! -name "*.txt"
find /home/ -mtime -1 \! -type d
find $HOME -mtime -1
find $HOME -mtime -7
find $HOME -size -500b
find ~ -size -500b
find ~ -name 'arrow*'
find ~ -name 'arrow*.xbm'
find /home -type f -name *.sxw -atime -3 -user bruno
find ~ -name '*.xbm'
find $HOME -mtime +365
find /home -perm /u=r
find kat -printf "%f\n"
find /usr -newer /tmp/stamp$$
find /usr/ -path "*local*"
find /usr -newermt "Feb 1"
find . -path ./src/emacs -prune -o -print
find . -path "./sr*sc"
find . -maxdepth 1 -name "name1" -o -name "name2"
find . -name \*.h -print -o -name \*.cpp -print
find . -regex '.*\.\(cpp\|h\)'
find \( -name '*.cpp' -o -name '*.h' \) -print
find /dir/to/search -path '*/.*' -print
find /dir/to/search/ -name ".*" -print
find /home -name ".*"
find /home -type f -name "*.sxw" -atime -3 -user bruno
find . -name "*.html" -print
find . -name \*.html
find . -path "./foo" -prune -o -type f -name "*.html"
find . -path "./foo" -prune -o -path "./bar" -prune -o -type f -name "*.html"
find . -mtime 7 -name "*.html" -print
find . -mtime -7 -name "*.html" -print
find . -mtime +7 -name "*.html" -print
find /var/www -type f -name "*.html"
find /etc -exec grep '[0-9][0-9]*[.][0-9][0-9]*[.][0-9][0-9]*[.][0-9][0-9]*' {} \;
find /etc -type f -exec cat '{}' \; | tr -c '.[:digit:]' '\n' | grep '^[^.][^.]*\.[^.][^.]*\.[^.][^.]*\.[^.][^.]*$'
find . -iname '*.jar'
find src/js -name '*.js'
find . -name "*.js"
find dir1 -type f -a \( -name "*.java" -o -name "*.as" -o -name "*.xml" \)
find . -maxdepth 1 -mindepth 1 -iname '*.jpg' -type f
find . -name *.jpg -user nobody
find . -type f -iregex '.*\.jpe?g'
find /var/log -name "*.log" -print0
find $dir -type f -name $name -print
find . -type f
find . -maxdepth 3 -type f
find . -type f -empty
find FOLDER1 -type f -print0
find . -type f -name "*.php" ! -perm 644
find / -name "*.php"
find . -type f
find . \( \( -path "\.?.*" -type d \) -o -path "*normal*" \) -prune -o \( -type f \) -print
find . -type f -user tom
find "$dir" -maxdepth 1 -type f
find . -path "*.git" -prune -o -type f -print
find . -path "*.git*" -prune -o -type f -print
find . -type f -size +10k
find . -type f -size 10k
find . -type f -size -10k
find . -type f -newer file.log
find . -type f ! -perm 777
find . -type f -perm 777
find ${FOLDER} -type f ! -name \".*\" -mtime -${RETENTION}
find src/js -type f
find "$dir" -name "*.mod" -type f -print0
find $FILES_PATH -type f
find ./subdirectory/ -type f
find . -type f
find . -type f -name \*
find . -type f -print0
find . -mmin 60 -type f
find /root -type f -size +500M -printf "The %p file is greater than 500MB\n"
find . -type f
find . -mindepth 2 -type f
find . \( -name bbb -o -name yyy \) -prune -o -type f -print
find . -name mmm -prune -o -type f -print
find . -type f -amin +10
find . -type f -newer "$FILE"
find . -mtime 0 -type f
find / -type f -exec echo {} \;
find /home/user/demo -type f -perm 777 -print
find /path/ -type f -daystart -mtime +0
find ~/ -daystart -type f -mtime 1
find -L /target ! -type l
find . -type l
find ./ -type l
find . -name "*.tex"
find . -name \*.tex
find . -regex ".*\(\.txt\|\.pdf\)$"
find . -name "*.txt" -print
find ~ -name "*.txt" — print -o -name ".*" — print
find . \( -name "*.txt" -o -name "*.pdf" \)
find . \( -name skipdir1 -prune , -name skipdir2 -prune -o -name "*.txt" \) -print
find $1 -type f -name '*'$n'.txt'
find . -name "*.txt"
find . -name "*.txt" -printf "%f\n"
find -name “*.txt”
find . -name bin -prune -o -name "*.txt" -print
find . -type f -name "*.txt" ! -name README.txt -print
find . -mmin 0.5
find /home -name "*.txt"
find /home -iname "*.txt"
find /tmp -name *.txt
find $DBA/$ORACLE_SID/bdump/*.trc -mtime +7
find . -type f -group sunk
find . ! -user root
find -type f -name '*.ext' | grep -vFf list.txt
find . -type f -name \*.ext | xargs grep foo
find . -name '*.pdf' -or -name '*.PDF'
find . -size +10k -ls
find . -name '*.c' -ls
find /tmp/ -exec ls "{}" +
find | xargs ls
find -print0 | xargs -0 ls
find . -size 0 -ls
find /dir/to/search -path '*/.*' -ls
find /dir/to/search/ -type d -iname ".*" -ls
find /dir/to/search/ -name ".*" -ls
find $HOME -name ".*" -ls
find /dir/to/search/ -type f -iname ".*" -ls
find . -name 'my*' -type f -ls
find /home/ -type f -size +512k -exec ls -lh {} \;
find /home/ -type f -size 6579c -exec ls {} \;
find /home/peter -nouser -exec ls -l {} \; -ok chown peter.peter {} \;
find . -name "*.pl" -ls
find . -size -26c -size +23c -exec ls -l '{}' \;
find . -size -26c -size +23c -ls
find . -mtime -2 -type f -name "t*" -exec ls -l '{}' \;
find /usr/bin -type f -size -50c -exec ls -l '{}' ';'
find . -empty -exec ls -l {} \;
find /myfiles -exec ls -l {} ;
find / -dev -size +3000 -exec ls -l {} ;
find . -daystart -ctime 4 -ls -type f
find Música/* -type f -name ".*" -exec ls -l {} \;
find . -type l -exec ls -l {} \;
find . -name "*.txt" -exec ls -la {} \;
find . -name "*.txt" -exec ls -la {} +
find . -type d -ls | head
find . -name '*.deb' -printf "%f\n"
find . -printf 'Name: %f Owner: %u %s bytes\n'
find | head
find /tmp  | head
find /usr/local/apache/logs/ -type f -name "*_log"|xargs du -csh
find . -name "*.NEF" -exec basename \{\} .NEF \;
find . -name "*.flac" -exec basename \{\} .flac \;
find . -type f -exec echo chown username {} \;
find . -maxdepth 1 -name "*.jpg" -size -50k | xargs echo rm -f
find . -iname '*test*' -exec cat {} \;
find . -iname '*test*' -exec cat {} \;
find . -name  '*.txt' -exec cat {} \;
find . -type f | wc -l
find . -type d –print | wc -l
find /mount/point -type d | wc -l
find . -print | wc -l
find . -type f |wc -l
find . -name "*.html" -print | xargs -l -i wc {}
find | wc -l
find . -maxdepth 1 -type f |wc -l
find . -type f -empty | wc -l
find . -type f -not -empty | wc -l
find /home/you -iname "*.txt" -mtime -60 | wc -l
find /var -maxdepth 2 -type d -printf "%p %TY-%Tm-%Td %TH:%TM:%TS %Tz\n"
find / -type f -printf "\n%Ab %p" | head -n 11
find . -printf "%y %p\n"
find / -type f -size +20000k -exec ls -lh {} \; | awk '{ print $8 ": " $5 }'
find . -maxdepth 1 -name '[!.]*' -printf 'Name: %16f Size: %6s\n'
find . -size +100M -exec ls -s {} \;
find --help
find . -type f \( -name "*.htm*" -o -name "*.js*" -o -name "*.txt" \) -print0 | xargs -0 -n1 echo
find . -type f \( ! -iname ".*" \) -mtime +500 -exec ls {} \;
find . -type f -not -name ‘.*’ -mtime +500 -exec ls {} \;
find . -type f -name "*.txt" ! -path "./Movies/*" ! -path "./Downloads/*" ! -path "./Music/*" -ls
find ~/junk   -name "*" -exec ls -l {} \;
find /var/log/ -mtime +60 -type f -exec ls -l {} \;
man find
find . -type f -size +50000k -exec ls -lh {} \; | awk '{ print $9 ": " $5 }'
find /var/log -type f -size +100000k -exec ls -lh {} \; | awk '{ print $9 ": " $5 }'
find . -type d -maxdepth 1 -exec basename {} \;
find . -type d -maxdepth 1 -mindepth 1 -exec basename {} \;
find . -maxdepth 1 -name '*.dat' -type f -cmin +60 -exec basename {} \;
find . -prune -name "*.dat"  -type f -cmin +60 |xargs -i basename {} \;
find ./ -name "*.dat" -type f -cmin +60 -exec basename {} \;
find . -name "*.c" -exec wc -l {} \;
find . -name "*.c" -print | xargs wc -l
find . -name "*.c" -print0 | xargs -0 wc -l
find . -exec wc -l {} \;
find . -name '*' | xargs wc -l
find . -name "*.h" -print | xargs wc -l
find -name '*php' | xargs cat | wc -l
find /var/www/ -type f -name «access.log*» -exec du -k {} \;|awk '{s+=$1}END{print s}'
find . -type f -exec ls -s {} + | sort -n -r | head -3
find /etc/ -type f -exec ls -s {} + | sort -n | head -3
find . -xdev -printf ‘%s %p\n’ |sort -nr|head -20
find / -type f -print | xargs file
find --version
find --version
find -version
find / -type f -printf "\n%AD %AT %p" | head -n 11
find .
find .
curl -L -C - -b "oraclelicense=accept-securebackup-cookie" -O http://download.oracle.com/otn-pub/java/jce/8/jce_policy-8.zip
curl http://127.0.0.1:8000 -o index.html
curl http://example.com/textfile.txt -o textfile.txt
curl https://raw.github.com/creationix/nvm/master/install.sh | sh
curl https://www.npmjs.com/install.sh | sh
curl http://example.com/
curl -L https://get.scoop.sh
curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.31.0/install.sh | bash
find .
find . -name *.php -or -path "./vendor" -prune -or -path "./app/cache" -prune
true | sleep 10
bind -x '"\eW":"who"'
find . -print
tar -xzvf backup.tar.gz
$ tar xvfJ filename.tar.xz
tar -xvzf passwd.tar.gz
sudo tar xvf phantomjs-1.9.0-linux-x86_64.tar.bz2
gzip -dc archive.tar.gz | tar -xf - -C /destination
tar xzf archive.tar.gz -C /destination
tar xpvf /path/to/my_archive.tar.xz -C /path/to/extract
tar -zxvf $1
tac infile | sed -ne '/pattern2/,/pattern1/ p' | tac -
find / -perm 1551
join -t, file1 file2 | awk -F, 'BEGIN{OFS=","} {if ($3==$8 && $6==$9 && $7==$10) print $1,$2,$3,$4,$6,$7}'
join file{1,2}.txt | awk '$2 != $3 { print "Age of " $1 " is different" }'
find /usr/share \! -type d wc -l
bind -p | grep -a forward
find . -type f -name "*.java" -exec grep -il string {} \;
find . -iname foo -type d
find . -iname foo
find foo -path /tmp/foo/bar -print
find /tmp/foo -path /tmp/foo/bar -print /tmp/foo/bar
find foo -path /tmp/foo/bar -print
find foo -path /tmp/foo/bar -print
find foo -path foo/bar -print
find . -name '*.js' -\! -name 'glob-for-excluded-dir' -prune
find . -lname '*sysdep.c'
find ./ -regex "./cmn-.\.flac"
find . -type f -name "*.txt" -exec sed '/\-/s /\-.*//g'  {} \;
find . -name *.gif -exec ls {} \;
find -name '*.js' -not \( -path './node_modules/*' -o -path './vendor/*' \)
find /home/user/Desktop -name '*.bmp' -o -name '*.txt'
find /var/www -name *.gif -ctime +90 -ctime -180
find . \! -path "*CVS*" -type f -name "*.css"
find . -type f -name "*.java" -exec grep -l StringBuffer {} \;
find . -type f -name "*.java" -exec grep -il string {} \;
find /usr/local/doc -name '*.texi'
find foo -path foo/bar -print
find /opt /usr /var -name foo.scala -type f
diff <(curl -s http://tldp.org/LDP/abs/html/) <(curl -s http://www.redhat.com/mirrors/LDP/LDP/abs/html/)
find ./ -name "foo.mp4" -exec echo {} \;
find . \( -name a.out -o -name '*.o' -o -name 'core' \) -exec rm {} \;
find -daystart -mtime 2
find . -type d -name tmp -prune -o -print | cpio -dump /backup
find /home/archive -type f -name "*.csv"  -mtime -2 -exec gzip -9f {} \;
find . * | grep -P "[a-f0-9\-]{36}\.jpg"
find . \( -name a.out -o -name '*.' -o -name  'core' \) -exec rm {} \;
find . -name 'cmn-*.flac'
find . -name 'cmn-*.flac' -print | grep -P '[\x4e00-\x9fa5]'
find . -name 'cmn-*\.flac' -print | grep -P './cmn-[\x4e00-\x9fa5]\.flac'
find $HOME -name '*.c' -print | xargs grep -l sprintf
find . -name '*.js' -not -path '*exclude/this/dir*'
find . -name  "*.java"
find . -name  \*.java
find . -name "*.sh" -print0 | xargs -0 -I {} mv {} ~/back.scripts
find /users/tom -name '*.p[lm]' -exec grep -l -- '->get(' {} + | xargs grep -l '#hyphenate'
find . -name "*.css" -exec grep -l "#content" {} \;
find . -name "*.css" -exec sed -i -r 's/#(FF0000|F00)\b/#0F0/' {} \;
find / -user seamstress -iname “*.pdf”
find -name "*.cpp" -o -name "*.c"
find -regex '.*\.\(c\|cpp\)'
find . -type f \( -name "*.class" -o -name "*.sh" \)
find /home/pat -iname "*.conf"
find . -iregex "./[^/]+\.dat" -type f -cmin +60 -exec basename {} \;
find . -name "*.dat" -type f -cmin +60 | grep "^./[^/]\+dat" | sed "s/^.\///"
find . -type d
find . -type d -name "*"
find . -name '*foo*' ! -name '*.bar' -type d -print
find -type d
find . -type d -maxdepth 1
find . -type d -name build
find  / -type d -name "apt" -ls
find  / -type d -name "project.images"
find  / -type d -name "project.images" -ls
find /usr -name lib64 -type d|paste -s -d:
find / -type d -name root
find /home/john -type d -name test -print
find /tmp -type d -empty
find /tmp -type d -empty
find / -empty
find /tmp -type f -empty
find ~ -empty
find . -size 0
find / -executable
find  /home -type f -perm /a=x
find .  -type f  -exec ls -lrt {} \; |awk -F' ' '{print $9}'
find  / -name "apt"
find  / -name "apt" -ls
find . -atime +7 -size +20480 -print
find . -type f -name "*.tmp"  -exec rm -rf {} \;
find . -mtime -7
find . -printf "%i \n"
find . -type f -name "Foo*" -exec rm {} \;
find . -type f -name "*.js.compiled"
find ./js/ -name "*.js.compiled" -print0
find . -name "*.js.compiled" -exec rename -v 's/\.compiled$//' {} +
find . -name "*bsd*" -print
find /usr/bin | xargs file
find . -perm 777 -print
find . | xargs wc -l
find . -name some_pattern -print0 | xargs -0 -I % mv % target_location
find . -name some_pattern -print0 | xargs -0 -i mv {} target_location
find . -iregex ".*packet.*"
find . -size +1M -exec mv {} files \+
find . -size +1M -print0 | xargs -0 -I '{}' mv '{}' files
find . -size +1M -ok mv {} files \+
find ./ -size +1000k
find . -size +270M -size -300M
find . -size 300M
find . -size -300M
find . -size +300M
find . -amin 10
find /etc -daystart -ctime -1
find /etc -ctime -1
find ~ -newer /tmp/timestamp
find ~ -mtime 1 -daystart
find /home/bozo/projects -mtime 1
find -anewer /etc/hosts
find -newer /etc/passwd
find . ! -readable -prune
find . ! -perm -g+r,u+r,o+r -prune
find . -type f \( -name "*.js" ! -name "*-min*" ! -name "*console*" \)
find . -size -1c -print
find -size +2M
find . -size +4096k -print
find . -size -26c -size +23c -print
find . -type f -exec grep "applicationX" {} \;
find -mmin 1 -print
find -mmin 2 -print
find . -mmin +10 -print
find . -mtime +10 -print
find . -name "*,txt"
find . -name \? -mtime +0
find . -size +10M -size -50M -print
find . — size +10 -print
find . -type f -empty
find . -type f -size 0b
find -name '*macs'
find . -amin -30
find -newer /etc/passwd
find . -mtime -1 -print
find . -name \? -mtime -1
find . -atime +30 -print
find . -atime +7 -o -size +20480 -print
find ./ -daystart -ctime +2
find . -name \? -mtime +0
find . -perm -0002 -print
find . -name pro\*
find . -size -50k
find . -type f -perm 777 -exec chmod 755 {} \;
find / -size +100M
find / -iname "filename"
find / -newer /tmp/checkpoint
find / -nouser
find / -group users -iname "filename"
find / -user pat -iname "filename"
find / -atime 0
find / -nouser -nogroup
find / -atime +2
find / -size +3G
find / -size 2048c
find / -perm 777 -iname "filename"
find /myfiles -size 5
find /myfiles -atime +30
find /etc -newer /tmp/foo
find /path/to/dir -newermt yyyy-mm-dd ! -newermt yyyy-mm-dd -ls
find $HOME -mtime -1
find $HOME -mtime -7
find ~ -type f -mtime -2
find ~ -mtime 2 -mtime -4 -daystart
find / -name linux
find . -type f -not -name "*.html"
find . -amin -1
find . — name "[A‑Z]*" — print
find . -perm -600 -print
find . -perm +600 -print
find . -uid 0 -print
find . -type d ! -perm -111
find . -type f ! -perm -444
find -used +2
find / -name "dir-name-here"
find / -name filename -exec  nano '{}' \;
find / -name game
find /home -type f -mtime +90 -mtime -100  -exec rm  {} \;
find /usr -print
find /etc -type f -exec cat '{}' \; | tr -c '.[:digit:]' '\n' \ | grep '^[^.][^.]*\.[^.][^.]*\.[^.][^.]*\.[^.][^.]*$'
find /etc -exec grep '[0-9][0-9]*[.][0-9][0-9]*[.][0-9][0-9]*[.][0-9][0-9]*' {} \;
find /etc/sysconfig -amin -30
find ./ | grep -E 'foo|bar'
find . -type d \( -path dir1 -o -path dir2 -o -path dir3 \) -prune -o -print
find ! -path "dir1" ! -path "dir2" -name "*foo*"
find / -name *.mp3
find / -name *.mp3 -fprint nameoffiletoprintto
find . -name '*song*abc2009*.jpg' | sed 's/\(^.*song.*\)abc2009\(.*.jpg\)$/mv "&" "\1def2010\2"/' | sh
find /usr/bin -name [ef]*x
find / -type f ! perm 777
find / -name foo.bar -print
find / -name foo.bar -print -xdev
find / -name "*gif" -print
find /home/foo -name "*.gz"
find /nas/projects -name "*.h"
find . -type f -name ".*" -newer .cshrc -print
find ./ -type f -iregex ".*\.html$" -or -iregex ".*\.cgi$"
find . -name '*.java' -exec grep REGEX {} \;
find . -name '*.js'
find . -type f -name "*.JPG"
find . -name "*.jpg"
find */201111/* -name "*.jpg" | sort -t '_' -nk2
find /somepath -type f -iregex ".*\.(pdf\|tif\|tiff\|png\|jpg\|jpeg\|bmp\|pcx\|dcx)" ! -name "*_ocr.pdf" -print0
find build -not \( -path build/external -prune \) -not \( -path build/blog -prune \) -name \*.js
find build -not \( -path build/external -prune \) -name \*.js
find -name '*.js' -not -path './node_modules/*' -not -path './vendor/*'
find . -name '*.js' -not \( -path "./dir1" -o -path "./dir2/*" \)
find /home -type f -name *.log -size +100M -exec rm -f {} \;
find . -type f -exec wc -l {} +
find . -type f -print0 | xargs -0 wc -l
find . -type f -ls | awk '{print $(NF-3), $(NF-2), $(NF-1), $NF}'
find . -type f -print0 | xargs -0 sed -i '' 's/2013/2012/g'
find . -maxdepth 1 -type f -name '*~' -delete  -or -name '#*#' -delete
find . -maxdepth 1 -type f  -name '*~' -exec mv {} /tmp/ \;  -or  -name '#*#' -exec mv {} /tmp/ \;
find . -type f -print
find / -type f -iname "filename"
find / -executable
find / -readable
find . -name “*.pdf” -print
find /var/www/ -type f -iname "*.php" -print
find . -type f -name '*.png'
find . -regextype posix-extended -regex "[a-f0-9\-]\{36\}\.jpg"
find . -type f -iname '*.CR2' -print0 | xargs -0 -n 1 -P 8 -I {}
find /usr/share/doc -name README
find / -readable
find /usr/share/doc -name README
find . -regextype sed -regex ".*/[a-f0-9\-]\{36\}\.jpg"
find . -regex '\./[a-f0-9\-]\{36\}\.jpg'
find . -regex "./[a-f0-9\-]\{36\}\.jpg"
find . -type f -name "*html" | xargs tar cvf jw-htmlfiles.tar -
find . -type f -print0| xargs -0 grep -c banana| grep -v ":0$"
find .  \( ! -path "./output/*" \) -a \( -type f \) -a \( ! -name '*.o' \) -a \( ! -name '*.swp' \) | xargs grep -n soc_attach
find . -type f -exec sed -i 's/searc/replace/g' {} \;
find /etc/sysconfig -amin -30 -type f
find ~ -type f -mtime -2
find -type f -mtime -60
find -type f
find -mtime -5 -type f -print
find . \( -iname "*photo*" -or -name "*picture*" \) -and ! -type d -and -mmin -30
find /myfiles -type f -perm -o+rw
find . -type f -path "*/cpp/*"
cp `find -perm -111 -type f` /usr/local/bin
find ./ -name "*.sqlite"
find -type l
find /usr -type l
find /home/john -name "*.sh" -type f -print
find . -name ".txt" | grep "foo" | xargs rm
find . -name ".txt" -exec rm "{}" \;
find . -name ".txt" | grep a
find . -type f -name '*.txt' -exec sed --in-place 's/[[:space:]]\+$//' {} \+
find . -iname '*.txt' -type f -exec sed -i '' 's/[[:space:]]\{1,\}$//' {} \+
find . –name "*.txt" –mtime 5
find . -name "*.txt" -print
find . -name '*.txt' | cpio -pdm /path/to/destdir
find . -name "*.txt" -print
find -name "*.txt"
find . -name '*.txt' \! -wholename '*/.svn/*' -exec grep 'sometext' '{}' \; -print
find . -name "*.txt" -newer /tmp/newerthan
find FFF -name "*.txt" -exec md5sum '{}' \;
find /user/directory/* -name "*txt" -mtime 0   -type f -exec awk '{s=$0};END{print FILENAME, ": ",s}' {} \;
find /user/directory/ -name "*txt" -mtime 0 -type f -printf '%p: ' -exec tail -1 {} \;
find . -name '*.text' -exec $SHELL -c '[ ! -f ${1%.*} ]' $SHELL '{}' ';' -print
find ./ -name '*.JPG' -type f | wc -l
find /home/you -iname "*.c" -atime -30 -type -f
find /home/david -mtime -2 -name '*.c'
find ./ -name '*.jpg' -type f | wc -l
find -iname *.SH
find "${current_directory}" -type f -iname "*.wma"
find /home -size +5000000b -name "error_log" -exec rm -rf {} \;
find /win/C -iname *JPG
find / -iname passwd
find / -name "*.conf"
find / -type f -name *.jpg  -exec cp {} . \;
find . -type f -name "*.bak" -exec rm -f {} \;
find . -type f -name “FILE-TO-FIND” -delete;
find . -type f -name "*.bak" -exec rm -i {} \;
find /home/pat -iname "*.conf"
find /etc -name "*.conf" -printf "%f %a, %t\n"
find / -name "*.conf"
find /etc/sysconfig -amin -30
find . -type f \( -name "*.conf" -or -name "*.txt" \) -print
find / -name core -exec rm -f {} \;
find /tmp -name core -type f -print | xargs /bin/rm -f
find -name "*.cpp"
find . -iname '*.cpp' -print
find . -type f -iname '*.cpp' -exec mv {} ./test/ \;
find . -type f -iname '*.cpp' -exec mv -t ./test/ {} \+
find . -name *.cpp -o -name *.h -o -name *.java
find -name '*.css'
find /var/named -type f -name *.db
find -type d
find -mindepth 3 -type d ! -path '*/p/*' -name c -print
find -mindepth 3 -type d -path '*/p/*' -prune -o -name c -print
find . -type d -delete
find . -maxdepth 1 -type d -print0
find . -type d -name "test*"
find $LOGDIR -type d -mtime +0 -exec compress -r {} \;
find -type d ! -perm -111
find -type d
find . -type d -print
find -type d | ssh server-B 'xargs -I% mkdir -p "/path/to/dir/%"'
find -type d|sed -r '/^.$/{s:.:#!/bin/bash:};{s/^\./mkdir -p &/}'
find . -maxdepth 1 -type d | xargs -I X mkdir '/new/directory/X'
find -type d -empty
find . -regex './kt[0-9] '
find . -depth -type d -mtime 0 -exec mv -t /path/to/target-dir {} +
find . -type d -mtime -0 -exec mv -t /path/to/target-dir {} +
find . -type d -mtime -0 -print0 | xargs -0 mv -t /path/to/target-dir
find . -type d -mtime 0 -exec mv {} /path/to/target-dir \;
find /media/New\ Volume/bhajans -maxdepth 1 -type d | xargs -0 mkdir -p
find / -type d -size +50k
find / -type d -name 'man*' -print
find / \( -type d -a -perm -777 \) -print
find $LOGDIR -type d -mtime +5 -exec rm -f {} \;
find /raid -type d -name ".local_sd_customize" -print
find /home -maxdepth 1 -type d
find /somePath -type d -name ‘someNamePrefix*’ -mtime +10 -print | xargs rm -rf ;
find . -type d -name “DIRNAME” -exec rm -rf {} \;
find ./ -maxdepth 1 -name "some-dir" -type d -print0 | xargs -0r mv -t x/
find . -type d -name 'uploads'
find ./ -empty -type d -delete
find . -type d -empty
find . -depth -empty -type d
find / -empty
find . -maxdepth 1 -type d -empty
find . -type f -empty -delete
find /opt \( -name error_log -o -name 'access_log' -o -name 'ssl_engine_log' -o -name 'rewrite_log' -o  -name 'catalina.out' \) -size +300000k -a -size -5000000k
find ./ -daystart -mtime -3
find ./ -daystart -mtime -3
find ./ -daystart -mtime +3
find . -name '*.[ch]' -print0 | xargs -r -0 grep -l thing
find . -name '*.[ch]' | xargs grep -l thing
find /opt -cmin -120
find / -ctime -1
find / -mtime -1
find / -daystart -mtime +41 -mtime -408 \   -printf "%M %n %u %g %10s %TY-%Tm-%Td %Ta %TH:%TM:%TS %h/%f\n" | awk '($7=="Fri"){print}'
find . -name '*.coffee' -exec awk '/re/ {print;exit}' {} \;
find . -name \*.coffee -exec grep -m1 -i 're' {} \;
find /usr -name '*.foo' -print
find /dir \( -name foo -prune \) -o \( -name bar -prune \) -o -name "*.sh" -print
find . -name '*.clj' -exec grep -r resources {} \;
find . -name *.clj | xargs grep -r resources
find -name "*.mkv"
find . -name "*rb" -print0 | xargs -0 head -10000
find -name '*~' -print0 | xargs -0 -I _ mv _ /tmp/
find . -name *.ini
find . \( -name '*.mp3' -o -name '*.jpg' \) -print
find ./ -iname '*.jpg' -type f | wc -l
find ./ -type f -regex ".*\.[Jj][Pp][gG]$" | wc -l
find . -name "*.undo" -ls | awk '{total += $7} END {print total}'
find / \( -nogroup -o -noname \) -ls
find . -size 6M
find . -size +10M -size -20M
find . -size +2G
find . -size -10k
find . -name *.ini -exec grep -w PROJECT_A {} \; -print | grep ini
find ./ -type f -regex ".*\.[JPGjpg]$" | wc -l
find . -mmin -60
find $HOME -mtime -1
find /usr -newer /usr/FirstFile -print
find /usr ! -newer /FirstFile -print
find . -exec chmod 775 {} \;
find . -exec echo {} ;
find . -print -o -name SCCS -prune
find . -exec grep -i "pw0" {} \;
find -cnewer /etc/fstab
find -mmin -60 -exec ls -l {} \;
find . -name ".*\.i"
find . -name *.bar -maxdepth 2 -print
find . -name "*bash*"
find . -wholename '/lib*'
find . -size +1000M
find . -name '*.orig'  -exec echo {} \ ;
find . -atime +1 -type f -exec mv {} TMP \;
find . -newer file
find . -atime +6
find . -mtime 0
find . -atime +30 -exec ls \;
find . -inum 31246 -exec rm [] ';'
find . -size -40 -xdev -print
find . -mmin -720 -mmin +60 -type f -name "t*" -exec ls -l '{}' \;
find . -size 126M
find . -xdev -name "wagoneer*" -print
find . -print0 | xargs -0 -l -i echo "{}";
find . -exec echo -n '"{}" ' \;
find $PWD -exec echo -n '"{}" ' \; | tr '\n' ' '
find $PWD | sed -e 's/^/"/g' -e 's/$/"/g' | tr '\n' ' '
find . | sort
find . | grep -vf file.lst
find . -exec grep -i "vds admin" {} \;
find . -newer document -print
find -name *bar
find . -name \? -daystart -mtime +0 -mtime -3
find -atime 0
find -daystart   -atime 0
find ./ -mtime 3
find ./ -mtime -3
find -mtime -7 -daystart
find ./ -mtime +3
find . -size +10M -size -50M -print
find . — size +1000k -print
find . -size +9M
find . -size 1234c
find . -newer /bin/sh
find . -readable
find . -size -9k
find . -writable
find . -user root -perm -4000 -print
find . -nouser -ls
find . -type d ! -perm -111
find . -type f ! -perm -444
find -E . -regex ".*ext1|.*ext2|.*ext3"
find . ( -name a.out -o -name *.o ) -print
find . -perm -4000 -print
find . -type f -atime 1
find . -amin -60
find . -cmin -60
find . -newer disk.log -print
find . -mtime +30 -a -mtime -7 -print0
find -mmin +15 -mmin -25
find . -mmin -60
find . -mindepth 1 -mmin -60 | xargs -r ls -ld
find . -daystart -mtime -1 -ls
find . -type f -mmin 0
find . -perm /a=x | head
find . -executable
find . -perm /u=x,g=x,o=x
find . -mtime +90 -type f -exec rm -f {} \;
find . -perm /g+s | head
find . -regex '.*\(net\|comm\).*'
find . -name 'cache*' -depth -exec rm {} \;
find . -name \? -daystart -mtime +0 -mtime -3
find . — name "[a‑z][a‑z][0—9][0—9].txt" — print
find ~ -type f -name test-a -exec mv {} test-10 \;
find ~ -type f -name test-a -execdir mv {} test-10 \;
find $1 \( -name "*$2" -o -name ".*$2" \) -print
find / -path /proc -prune -o -nouser -o -nogroup
find / -name 'top?????*'
find / -atime 50
find / -amin -60
find / -cmin -60 | head
find / -mtime +50 -mtime -100 | head
find / -mtime 50
find / -name "*conf" -mtime 7
find / -name 'top???'
find / -name 'top*'
find / -mmin -10
find / -group staff -print
find / -user roger -print
find / -mtime -5 -print
find / -mtime -2 -print
find / -mtime -5 -print
find / -size +50M -size -100M
find / -size 15M
find / -type f -size +20000k
find / -group root | head
find / -user root | head
find / -user www -print
find / -mount -name 'win*'
find / -mtime -1
find / -size +3 -print
find / -group users -iname "Dateiname"
find / -user pat -iname "Dateiname"
find /    \( -perm -4000 -fprintf /root/suid.txt '%#m %u %p\n' \) , ( -size +100M -fprintf /root/big.txt '%-10s %p\n' \)
find / -atime -1
find / -newerct '1	minute ago' -print
find / -ctime -1
find / -mmin -10
find / -mtime -30 -print
find / -mmin -60
find / -mtime +100 -print
find / -perm /u=r | head
find / -perm -u+s
find / -perm 777 -iname "Dateiname"
find / -nogroup -print
find /mp3-collection -size +10000k ! -name "Metallica*"
find /u/bill -amin +2 -amin -6
find /usr/app/etl/01/OTH/log/tra -type f ! -name ".*" -mtime -10 | egrep -vf /usr/app/etl/01/CLE/par/files_to_skip.par
find /myfiles -mtime 2
find /myfiles -atime +30
find ./machbook -exec chown 184 {} \;
find /home/calvin/ -mmin -45
find /home -size +10M -size -50M
find /home -size 10M
find /opt -atime 20
find /opt -mtime +30 -mtime -50
find /opt -mtime 20
find /path/to/dir -newermt “Feb 07”
find /usr/bin -type f -mtime -10
find /work -user olivier -print
find Música/* | egrep -Z \/\\. | xargs -0 echo
find ~/Music/ -name "Automatically Add*"
find $HOME -mtime -2 -mtime +1
find $HOME -mtime -2 -mtime +1
find $HOME -mtime +365
find ~ -size +2000000c -regex '.*[^gz]' -exec gzip '{}' ';'
find ~ -empty
find ~ -size -300b
find / -size 42
find ~ -iname '*.tex'
find $HOME -newer ~joeuser/lastbatch.txt
find $HOME -mtime +365
find $HOME -mtime -1
find $HOME -mmin -30
find $HOME -mtime -7
find ~ -type f -mtime 0
find $HOME -mtime +365
find `pwd` -group staff -exec find {} -type l -print ;
find / -mtime -1 -print
find /tmp -mtime +30 -print
find Symfony -name '*config*';
find Symfony -iname '*config*';
find -daystart -mtime 1
find  -mtime -1
find -mtime +0 -mtime -1
find -daystart -mtime -7
find -daystart -mtime +7
find . -ctime 1 -type f
find . -ctime 0 -type f
find -mtime 1
find -mtime 2
find . -type f -mtime 1
find . -type f -daystart -mtime 1
find . -mtime 7
find . -type f -daystart -mtime -1
find -mtime -1
find ./ -mtime -0.5
find -daystart -mitime -1
find . -type f -daystart -mtime -2
find . -type f -mtime -1
find . -mtime 4 -daystart -exec cp -a {} /home/devnet/fileshare\$ on\ X.X.X.X/RECOVER/ \;
find -mmin -60
find . -mtime -7 -print
find /etc -newer /var/log/backup.timestamp -print
find . -type f -mtime 0
find /tmp/test/* -mtime +0
find . -type f -daystart -mtime 0
find . -mtime +7
find -mtime +2
find /tmp/test/* -mtime +1
find /etc -name *fstab*
find . -daystart -ctime 1 -type f
find . -daystart -ctime 0 -type f
find /usr/local -mtime 1
find . -type f -empty
find . -name '*.deb' -exec basename {} \;
find . -name '*.deb' | xargs -n1 basename
find -anewer /etc/hosts
find -cnewer /etc/fstab
find . -mmin -15 \( ! -regex ".*/\..*" \)
find . -type f -mtime +356 -printf '%s\n'  | awk '{a+=$1;} END {printf "%.1f GB\n", a/2**30;}'
find /tmp -type f -empty
find . -perm -20 -exec chmod g-w {} ;
find /mp3-collection -name 'Metallica*' -and -size +10000k
find / -size +50M -iname "Dateiname"
find -iname september
find . -iname test
find ~ -atime 100
find / -name findme.txt -type f -print
find / -name .ssh* -print | tee -a ssh-stuff
find . -name "foo.txt" | awk '{ print "mv "$0" ~/bar/" | "sh" }'
find usr/include -name '*.h' -mtime -399 | wc
find /usr/include -type f -mtime -400 -name "*.h"
find /tmp -type f -name ".*"
find . -name "*.html"
find ./ -type f -name '*.html' | xargs sed -i '1,/sblmtitle/d'
find -name '*.html' -print0 | xargs -0 rename 's/\.html$/.var/'
find ./ -type f -name '*.html' | xargs sed -i '$s/$/<\/description>/'
find . -mtime 7 -name "*.html" -print
find . -mtime 7 -name "*.html" -print
find . -mtime -7 -name "*.html" -print
find . -mtime +7 -name "*.html" -print
find . -mtime 1 -name "*.html" -print
find . -mtime -7 -name "*.html"
find . -type f -name "*.htm*" -o -name "*.js*" -o -name "*.txt"
find . -regex '.+\.js'
find . -type f|grep -i "\.jpg$" |sort
find /ftp/dir/ -size +500k -iname "*.jpg"
find "somedir" -type l -print0
find . -type l -print | xargs ls -ld | awk '{print $10}'
find -L /target -type l
find /target -type l -xtype l
find / -name "*.log"
find / -xdev -name "*.log"
find . -name "*.mp3" -exec mv {} "/Users/sir/Music//iTunes/iTunes Media/Automatically Add to iTunes.localized/" \;
find / -type f -name *.mp3 -size +10M -exec rm {} \;
find / -iname "*.mp3" -print
find ~ -type f -mtime 0 -iname '*.mp3'
find . \! -name "*.Z" -exec compress -f {} \;
find $HOME -type f -atime +30 -size 100k
find . -type f -size +10000 -exec ls -al {} \;
find /etc/sysconfig -amin -30 -type f
find . -type f -print0 | xargs -0 grep pattern
find -type f
find . -type f -exec grep -il mail
find . -mtime -1 -type f -print
find . type f -print | fgrep -f file_list.txt
find . -type f -atime -1 -exec ls -l {} \;
find . -type f -mtime -1 -exec ls -l {} \;
find . -type f -mtime -1 -daystart -exec ls -l {} \;
find . -type f -mtime 2 -mtime -3 -daystart -exec ls -l {} \;
find pathfolder -maxdepth 1 -type f -not -path '*/\.*' | wc -l
find pathfolder -mindepth 2 -maxdepth 2 -type f -not -path '*/\.*' | wc -l
find main-directory -type f
find -name *monfichier*.ogg
find -name '*.patch' -print0 | xargs -0 -I {} cp {} patches/
find . -type f -name '*.pdf' |sed 's#\(.*\)/.*#\1#' |sort -u
find /var/www/ -type f -name "*.pl" -print
find /var/www/ -type f -iname "*.pl" -print
find . -type f -name "*.pl"
find . -name '*.pl' | xargs grep -L '^use strict'
find . -type f -name "*.pl" -print0
find -name '*.php'
find -name '*.php' -exec grep -li "fincken" {} + | xargs grep -l "TODO"
find -name '*.php' -exec grep -in "fincken" {} + | grep TODO | cut -d: -f1 | uniq
find . -regex '.+\.php'
find . -name \*.php
find . -name “*.[php|PHP]” -print
find . -name \*.php -type f
find . -regex '.+\.\(php|js\)'
find . -name '*.png' | grep -f <(sed 's?.*?/[0-9]_[0-9]_[0-9]_&_?' search.txt)
find . -name '*.png' | grep -f <(sed s/^/[0-9]_[0-9]_[0-9]_/ search.txt)
find . -name '*.png' | grep -f <(sed s?^?/[0-9]_[0-9]_[0-9]_? search.txt)
find . -name '*.png' | grep -f <(sed s?^?/[0-9]_[0-9]_[0-9]_? search.txt) | xargs -i{} cp {} /path/to/dir
find . -name '*.png' | grep -f search.txt
find . -name "image*.png"
find /home/pankaj -maxdepth 1 -cmin -5 -type f
find . -type f -name "*.php"
find . -type f -ctime -3 | tail -n 5
find -type f ! -perm -444
find . -type f -name '*some text*'
find /some/dir -type d -exec find {} -type f -delete \;
find /path -type f -delete
find /path -type f -exec rm '{}' \;
find /path -type f -print0 | xargs -0 rm
find . -type f -atime +30 -print
find * -type f -print -o -type d -prune
find . -type f \! -name "*.Z" \! -name ".comment" -print | tee -a /tmp/list
find -type f
find . -maxdepth 1 -type f
find main-directory -type f -exec mv -v '{}' '{}'.html \;
find . -type f | sed -e 's#.*\(\.[a-zA-Z]*\)$#\1#' | sort | uniq
find . -group flossblog -type f
find . -user sedlav -type f
find . -uid +500 -uid -1000 -type f
find /myfiles -type f -perm -647
find /travelphotos -type f -size +200k -not -iname "*2015*"
find . -type f \( -name "*.sh" -o -name "*.pl" \)
find . -name "*~" -delete
find . -name "*~" -exec rm {} \;
find /etc -type l -print
find ./ -type l -exec file {} \; |grep broken
find -L . -type l
find -L
find . -xtype l
find . -name somedir -prune , -name bin -prune -o -name "*.txt" -print
find . \( -name somedir -prune \) , \( -name bin -prune \) -o \( -name "*.txt" -print \)
find . -name "*.txt" | xargs rm -rf
find . -name "*.txt" | xargs -I '{}' mv '{}' /foo/'{}'.bar
find . -name "*.txt" -type f -daystart -mtime -4 -mtime +0|xargs -i cp {} /home/ozuma/tmp
find . -name "*.txt" -print
find . -type f -name '*.txt' -print
find . — name "*.txt" — print
find . -name "*.txt" -print | less
find . -name "*.txt" -printf "%M %f \t %s bytes \t%y\n"
find -maxdepth 1 -iname "*.txt"
find . -path "./sk" -prune -o -name "*.txt" -print
find . -name "somefiles-*-.txt" -type f
find . -name "somefiles-*-.txt" -type f -exec sed -i 'iText that gets prepended (dont remove the i)' -- '{}' \;
find / -user root -iname "*.txt" | head
find / -mount -name "*.txt"
find / -xdev -name "*.txt"
find /home/calvin/ -maxdepth 2  -name “*.txt”
find /home/calvin/ -mindepth 2  -name “*.txt”
find ~/ -name '*.txt'
find ~ -name "*.txt" — print
find /tmp -type f -name ‘*.txt*’ | sed -e ‘s/.*/\”&\”/’ |xargs -n 1 grep -l hello|sed -e ‘s/.*/\”&\”/’
find . -type f -name "*.txt" ! -path "./Movies/*" ! -path "./Downloads/*" ! -path "./Music/*"
find . -name "*.txt" -type f -daystart -mtime +0 -mtime -2
find . -type f \( -iname "*.txt" ! -iname ".*" \)
find ./ -name *.undo | xargs wc
find ~ -type f -exec file -i {} + | grep video
find /tmp /var/tmp ~ -type f -size +10M -mtime +60 -ctime -100 -exec file -N -i -- {} + | sed -n 's!: video/[^:]*$!!p'
find . -name '*.wav' -maxdepth 1
find /var/www/ -name wp-config.php
find /var/www/ -name wp-config.php -maxdepth 2
find . -name "*.xml" -exec grep -HFf /tmp/a {} \;
find . -name \*.xml | grep -v /workspace/ | tr '\n' '\0' | xargs -0 tar -cf xml.tar
find . -name "*.xml" -exec grep -HFf <(find . -name "*.txt" -printf "%f\n") {} \;
find . -type f -name '*.zip'
find . -type f -name '*.zip' -print0 | xargs -0 tar -xzf
find -name "*.js" -not -path "./directory/*"
find . -path ./misc -prune -o -name '*.txt' -print
find . -depth -empty -type d -delete
find / -size +100M -exec rm -rf {} \;
find . -iname "Articles.jpg"
find . -regex './[0-9].*' -print
find . -iname .svn -exec bash -c 'rm -rf {}' \;
find -iname example.com | grep -v beta
find ./ -path ./beta/* -prune -o -iname example.com -print
find . -mtime -7 -type d
find . -mtime -7 -type d
find /usr/spool/uucp -type d -print
find $LOGDIR -type d -mtime +0 -exec compress -r {} \;
find $LOGDIR -type d -mtime +5 -exec rm -f {} \;
find . -path './bar*' -print
find . -iname foo -type d
find /users/al -name Cookbook -type d
find . -name "*.txt"
find . -name foo.txt
find / -name foo.txt
find . -iname foo
find . -name "foo.*"
find / -atime -1
find . -mtime -1 -type f
find . -name '*.jpg' -print ./bar/foo.jpg
find . -name "*.bam"
find $HOME \( -name \*txt -o -name \*html \) -print0 | xargs -0 grep -li vpn
find /dir/path/look/up -name "dir-name-here"
find /dir/path/look/up -name "dir-name-here" -print
find /tmp -name core -type f -print | xargs /bin/rm -f
find /tmp -name core -type f -print0 | xargs -0 /bin/rm -f
find /u/bill -amin +2 -amin -6
find /usr -newermt "Feb 1"
find . -name game
find . -name "*.[ch]" -exec grep --color -aHn "e" {} \;
find . -name "S1A*1S*SAFE" | awk -F/ '{print $NF"/"$0}' | sort -t_ -k 5,5 | cut -d/ -f 2-
find . -name "S1A*1S*SAFE" | rev | awk -F '/' '{print $1}' | rev | sort -t _ -k 5
find ~ -atime 100
find ~ -name game
find ~/ -daystart -type f -mtime 1
find / -name game
find /usr/src -name '*.c' -size +100k -print
find . -cmin -60
find -amin -60
find . -mmin -60
find -iname "filename"
find . -name '*.[ch]' | xargs grep -l thing
find . -name '*.[ch]' -print0 | xargs -r -0 grep -l thing
find ~/ -daystart -type f -mtime 1
find $HOME/. -name *.txt -ok rm {} \;
find . \( -name "foo" -o -name "bar" \)
find . -name "* *" -exec rm -f {} \;
find . -name '*.txt' -print -o -name '*.html'
find . -type f ! -perm 777 | head
find /tmp/foo -path /tmp/foo/bar -print
find /tmp/foo -path /tmp/foo/bar -print
find .  -path '*/*config'
find .  -path '*f'
find . -type f -perm 0777 -print
find . -mtime -7
find . -mtime 1
find . -name '*.h' -execdir diff -u '{}' /tmp/master ';'
find /etc -name '*.conf'
find . -iname foo
find . -iname foo -type d
find . -iname foo -type f
find . -name "photo*.jpg"
find . -type f -exec grep -li '/bin/ksh' {} \;
find . -type f -print | xargs grep -li 'bin/ksh'
find /var -name lighttpd
find . -print|grep sql|xargs grep -i dba_2pc_pending
find ./ -regex "cmn-.*[\x4e00-\x9fa5]*\.xml"
find /etc -name "httpd.conf"
find $HOME -name \*txt -o -name \*html -print0
sudo find / -name mysql -print
find / -perm 0551
find / -type d -name 'httpdocs'
find . -exec ls -ld {} \;
echo 'string to be hashed' | md5
md5 -s 'string to be hashed'
yosemite$ echo -n 401 | md5
echo -n '' | md5
curl -s www.google.com | md5
echo -n hi | md5
md5sum file*.txt
find . -mmin -15 \( ! -regex ".*/\..*" \)
find  / -type d -iname "apt"
find  / -type d -iname "apt" -ls
find  / -type d -iname "project.images" -ls
groups user
find / -name foo.txt -type f
find / -name foo.txt -type f -print
find . -mtime -7 -type f
find ~/mail -type f | xargs grep "Linux"
find -maxdepth 1 -type f -printf '%f\000'
find -type f -exec md5sum {} +
find . -type f -wholename \*.mbox | sed 's/\(.*\)\.mbox/mv "\1.mbox" "\1"/' | sh
find . -wholename \*.mbox | awk '{new=$0; gsub("\.mbox$", "", new) ; system("mv \"" $0 "\" \"" new "\"") }'
find . -mtime -7 -type f
find . -iname foo -type f
find -type f
find / \( -perm -4000 -fprintf /root/suid.txt '%#m %u %p\n' \) , \  \( -size +100M -fprintf /root/big.txt '%-10s %p\n' \)
find ./n* -name "*.tcl"
find . -lname '*sysdep.c'
find -iname "MyCProgram.c"
find -iname "MyCProgram.c" -exec md5sum {} \;
find . -type f -exec du -Sh {} + | sort -rh | head -n 15
find . -type f | xargs | wc -c
find /usr -type f | wc -l
find . -maxdepth 1 -name \*.txt -print0 | grep -cz .
find folder1/ -depth -type f -printf "%d\t%p\n"
find -iname "MyCProgram.c"
find -iname "Dateiname"
find / -iname "Dateiname"
find / -name filename.txt -print
find /usr -name filename.txt -print
OUTPUT=`find . -name foo.txt`
find / -type f -name httpd.log
find /home/web-server/ -type f -name httpd.log
find /home/web-server/ -type f -iname httpd.log
find /home/user/myusername/ -name myfile.txt -print
find / -name arrow.jpg
find . -inum $inum -exec rm {} \;
find -print | grep esxcfg-firewall
find / -name file
find teste1 teste2 -type f -exec md5 -r {} \; | sort
find . -type f -printf '%TY-%Tm-%Td %TT   %p\n' | sort
find -type f -printf '%T+ %p\n' | sort | head -n 1
find . -type f -print0 | xargs -0 ls -ltr | head -n 1
find -type f -printf "%T+ %p\0" | sort -z | grep -zom 1 ".*" | cat
find ! -type d -printf "%T@ %p\n" | sort -n | head -n1
find . -name foo.mp4 | sed 's|/[^/]*$||'
find ./ -name "foo.mp4" -printf "%h\n"
find . ! -path "*/test/*" -type f -name "*.js" ! -name "*-min-*" ! -name "*console*"
find /root/ -name myfile -type f
find /home -type f -exec du -s {} \; | sort -r -k1,1n | head
find . -type f -exec ls -al {} \; | sort -nr -k5 | head -n 25
find /home -type f -print0 | xargs -0 file
find . -type f -exec file {} \;
find . -type f | xargs file
find . -type f -exec file {} \+;
find . -type f \( -iname ".*" ! -iname ".htaccess" \)
find ~ -type f -mtime 0 -ls
find . -type d -name CVS -exec rm -r {} \;
find . -type f -newermt "2013-06-01" \! -newermt "2013-06-20"
ln -f $GIT_DIR/../apresentacao/apresentacao.pdf $GIT_DIR/../capa/apresentacao.pdf
find $HOME -name core -exec rm -f {} \;
find 'Test Folder' -type d -print0 | xargs -0 rm -rf
find . | grep -v xml | xargs rm -rf {}
find . -mtime -3 -exec rm -rf {} \;
find /tmp/* -atime +10 -exec rm -f {} \;
find /home -type f -name test.txt -exec rm -f {} \
find . -type f -exec rm -fv {} \;
find /tmp -size 0 -atime +10 -exec rm -f {} \;
find . -name "*.c" | xargs rm -rf
find . -name "*.c" -print0 | xargs -0 rm -rf
find /var/www -type d -mtime 0 -name logs -exec sudo rm -fr {} \;
find /tmp -type f -name sess* -exec rm -f {} \;
find .  -name "*.txt" -type f -daystart -mtime +89 | xargs rm -f
ln -fs /etc/configuration/file.conf /etc/file.conf
ln -sfvn source target
ln -sfv /usr/local/opt/mongodb/*.plist ~/Library/LaunchAgents
sudo chown -Rf www-data *
ln -sf new_destination linkname
find . -type f -not -name "*.html"
jobs -x echo %1
jobs -l | grep 'test.sh &' | grep -v grep | awk '{print $2}'
find . -mtime -7 -print0 | xargs -0 tar -cjf /foo/archive.tar.bz2
find . -mtime -7 -print0 | xargs -0 tar -rf /foo/archive.tar
find . -mtime 30 -print
find . -mtime -30 -print
md5=`md5sum ${my_iso_file} | awk '{ print $1 }'`
cat file.txt | rev | cut -d ',' -f 2 | rev
find . -atime +30 -exec ls \; | wc -l
su git
sudo su
sudo su -
sudo su
date +%Y-%m-%d
date +%Y-%m-%d:%H:%M:%S
head -5 tst.txt | tail -1 |cut -c 5-8
tac a | grep -m1 -oP '(?<=tag>).*(?=</tag>)'
tac your.log | grep stuff
bind '"e":self-insert'
find /path/to/dir ! -perm 0644 -exec chmod 0644 {} \;
find /path/to/dir/ -type f ! -perm 0644 -print0 | xargs -0 chmod 644
jobs -p | tail -n [number of jobs] | xargs kill
jobs -p | xargs kill -9
jobs -p | xargs kill
kill -INT $(jobs -p)
kill $(jobs -p)
kill `jobs -lp`
cat -n text.txt | join -o2.2 lines.txt -
find /home/user/Desktop -name '*.bmp' -o -name '*.txt'
find /home/user/Desktop -name '*.pdf'
find /home/user/Desktop -name '*.pdf' -o -name '*.txt' -o -name '*.bmp'
jobs -lp
find src -name "*.java"
find ~/ -name '*.txt'
find . -type f \( -name "*.c" -o -name "*.sh" \)
find . -name "*.css"
find . -type f -name "*.css"
jobs -l
find .
find /home/bozo/projects -mtime 1
find /home/bozo/projects -mtime -1
find .
find -name cookies.txt
find . -type f -name "*.java" -exec grep -l StringBuffer {} \;
find . -type d -name proc -prune -o -name '*.js'
find . -name '*.js' -and -not -path directory
find . -name '*.js' | grep -v excludeddir
find . -name '*.js' | grep -v excludeddir | grep -v excludedir2 | grep -v excludedir3
find /path/to/search                    \   -type d                               \     \( -path /path/to/search/exclude_me \        -o                               \        -name exclude_me_too_anywhere    \      \)                                 \     -prune                              \   -o                                    \   -type f -name '*\.js' -print
jobs -l
find $directory -type f -name '*'
find ! -path "dir1" ! -path "dir2" -type f
find dir -not \( -path "dir1" -o -path "dir2" -prune \) -type f
find dir -not \( -path "dir1" -prune \) -not \( -path "dir2" -prune \) -type f
jobs
find  /var -path */l??/samba*
find . -ls -name "*.ksh"
find httpdocs -type d
find -maxdepth 1 -type d
find . -empty
find / -path /proc -prune -o -perm -2 ! -type l -ls
find ./ -name "*.sqlite" -ls
find . -empty -exec ls {} \;
find . -newer /bin/sh
find "somedir" -type l -print0 | xargs -r0 file | grep "broken symbolic" | sed -e 's/^\|: *broken symbolic.*$/"/g'
find /proc/$1/exe -printf '%l\n'
find / -type d -gid  100
find . -name '*.txt' -o -name '*.html'
find /u/bill -amin +2 -amin -6
find /usr -newermt "Feb 1"
find /usr -newer /tmp/stamp$$
find "$directory" -perm "$permissions"
find . ! -perm -g+r,u+r,o+r -prune
find . ! -readable -prune
find / -mount \! -readable -prune  -o  -path /dev -prune  -o  -name '*.jbd' -ls
find / \! -readable -prune -o -name '*.jbd' -ls
jobs -l
bind -l | grep /
find . -type f \( -name "*.c" -o -name "*.sh" \)
find . -type f \( -name "*cache" -o -name "*xml" -o -name "*html" \)
find . -type f -not -name "*.html"
find . -type f -wholename \*.mbox
find . -type f
find . -type f -wholename \*.mbox -print0
find $HOME -type f -atime +30 -size 100k
find $directory -type l
find $directory -type l -printf "%p$IFS"
find . -type d -exec ls -ld {} \;
find . \( -name '*.mp3' -o -name '*.jpg' \) -name 'foo*' -print
find ~/ -daystart -type f -mtime 1
find . \( -name '*.txt'  -mtime +5 -o -name '*.html' \) -print0
find . \( -name '*.txt' -o -name '*.html' \) -mtime +5 -print0
find /home -type f -size +100M -delete
find /home -type f -size +100M -print0 |xargs -0 rm
sudo su -l oracle
su -l
find . -perm 777 -type f -exec ls -l {} \;
find . -perm 040 -type f -exec ls -l {} \;
find . -perm -g=r -type f -exec ls -l {} \;
find . -type f -name "Tes*" -exec ls -l {} \;
find . -name "*.sh" -exec ls -ld {} \;
find /home -type f -size +10485760c -print
mkdir /etc/cron.15sec
find /path/to/dir -type d -exec chmod 755 {} \;
find . -name "*" -maxdepth 1 -exec mv -t /home/foo2/bulk2 {} +
find . -mindepth 1 -exec mv -t /tmp {} +
find . -mindepth 1 -print0|xargs -0 -I, mv , /tmp
find . -follow -iname '*.htm' -print0 | xargs -i -0 mv '{}' ~/webhome
find . -atime +1 -type f -exec mv {} TMP \; # mv files older then 1 day to dir TMP
md5 -q file
find  / -type d -iname "apt" -ls
curl "url" | tac | tac | grep -qs foo
jobs -sl | awk '{print $2}'
find . -type f -exec grep -il 'foo' {} \;
find .
find . -print
find . -wholename './src/emacs' -prune -o -print
find / \! -name "*.c" -print
tac infile | sed '/string match/,$d' | tac
bind -P
du -B1 --apparent-size /tmp/foo.txt
bind -p|grep -i '"[pE]"'
du -sh *
tac INPUTFILE | sed '/^Statistics |/q' | tac
n_jobs=$( jobs -p | awk '{print NR}' )
bind -P | grep '\\e\\C-k'
tac a.txt | awk 'NF{print $NF; exit}'
tail -n "+$(grep -n 'TERMINATE' file | head -n 1 | cut -d ":" -f 1)" file
find /usr/local -type f -perm /a=x | xargs file |  grep 'not stripped' | cut -d: -f1
du -a . | sort -nr | head
tac | sed -n '/PATTERN/,+19{h;d};x;/^$/!{p;s/.*//};x' | tac
echo $filename | egrep -o '[[:digit:]]{5}' | head -n1
find `pwd` -maxdepth 1
head -n1 $bigfile
tac file.log | awk '{ if ($1 >= 423) print; else exit; }' | tac
tac $FILE | grep -m 1 '.'
tac FILE |egrep -m 1 .
tac file | sed -n '0,/<tag>\(.*\)<\/tag>/s//\1/p'
find /usr/src -name '*.c' -size +100k -print
history | awk '{print $2}' | awk 'BEGIN {FS="|"}{print $1}' | sort | uniq -c | sort -nr | head
find . -prune
history -n
bind '"\eY": "\e2\e."'
chown -R antoniod:antoniod /opt/antoniod/
sudo chown -R ec2-user:apache /vol/html
chown nginx:nginx /your/directory/to/fuel/ -R
chown -R user:www-data yourprojectfoldername
chown -R root:root /var/lib/jenkins
chown "dev_user"."dev_user" -R ~/.ssh/
sudo chown -R xxx /Users/xxx/Library/Developer/Xcode/Templates
sudo chown -R $USER ~/tmp
sudo chown -R $(whoami) /usr/lib/node_modules/
sudo chown -R `whoami` /usr/local
sudo chown -R `whoami` /usr/local/lib
sudo chown -R $USER /usr/local/lib/node_modules
chown -R nobody upload_directory
chown ftpuser testproject/ -R
sudo chown -R $(whoami) ~/.npm
sudo chown -R test /home/test
chown -R owner:owner public_html
find /data/bin/test -type d -mtime +10 -name "[0-9]*" -exec rm -rf {} \;
find /data/bin/test -type d -mtime +10 -name '[0-9]*' -print | xargs rm -rf ;
find . -type f -newermt "$(date '+%Y-%m-%d %H:%M:%S' -d @1494500000)"
find . -type f -regex ".*\.\(py\|py\.server\)"
find . \( -name "*.py" -o -name "*.py.server" \)
find .  -name '*.txt' -exec rsync -R {} path/to/dext \;
find . -type f -name "Foo*" -exec rm {} \;
find ~/ -name 'core*' -exec rm {} \;
find / -name "*.core" -print -exec rm {} \;
find / -name "*.core" | xargs rm
find /tmp -type f -mtime +30 -exec rm -f {} \;
find . -type d -name CVS -exec rm -r {} \;
find -name "*.txt" | xargs rm
find -name "*.txt" -print0 | xargs -0 rm
find / -name "*.core" -print -exec rm {} \;
find / -name "*.core" | xargs rm
find . -name core -ctime +4 -exec /bin/rm -f {} \;
find . -name .DS_Store -exec rm {} \;
find /usr -name core -atime +7 -exec rm "{}" \;
/bin/find -name "core" — exec rm {} \;
find /home -name core -exec rm {} \;
find /tmp -name core -type f -print0 | xargs -0 /bin/rm -i
find . -name ".DS_Store" -exec rm {} \;
find ./ -ctime +30 -type f -exec rm -f {} \;
find . -mtime +10 | xargs rm
find . -inum $inum -exec rm {} \;
find /myfiles -atime +30 -exec rm {} ;
find . -name "* *" -exec rm -f {} \;
find . -name '*[+{;"\\=?~()<>&*|$ ]*' -maxdepth 0 -exec rm -f '{}' \;
find /home -name Trash -exec rm {} \;
find /logs -type f -mtime +5 -exec rm {} \;
find . -name "*.pdf" -maxdepth 1 -print0 | xargs -0 rm
find * -perm 777 -exec chmod 770 {} \;
find /tmp -maxdepth 1 -type f -delete
rev urllist.txt | cut -d. -f 2- | rev
head --lines=-N file.txt
tac filename | sed 4,6d | tac
find . -name "*.png" -print0 | sed 'p;s/\.png/\.jpg/' | xargs -0 -n2 mv
find . -name "*.svg.png" -print0 | sed 's/.svg.png//g' | xargs -0 -I namePrefix mv namePrefix.svg.png namePrefix.png
find -name "*.txt" -exec mv {} `basename {} .htm`.html \;
find . -name "*.htm" -exec mv '{}' '{}l' \;
find -name ‘*exp_to_find_in_folders*’ -exec rename “s/exp_to_find_for_replacement/exp_to_replace/” {} \;
find . -type f -print0 | xargs -0 sed -i 's/Application/whatever/g'
find ./ -exec sed -i 's/apple/orange/g' {} \;
find ./ -type f -exec sed -i -e 's/apple/orange/g' {} \;
find . -type f -exec sed -i 's/foo/bar/g' {} +
find . -type f -name "*baz*" -exec sed -i 's/foo/bar/g' {} +
find . -type f -executable -exec sed -i 's/foo/bar/g' {} +
tac infile.txt | sed "s/a/c/; ta ; b ; :a ; N ; ba" | tac
tac | sed '0,/a/ s/a/c/' | tac
tac file | awk '/a/ && !seen {sub(/a/, "c"); seen=1} 1' | tac
tac file | sed '/a/ {s//c/; :loop; n; b loop}' | tac
tac file | sed '2 s/,$//' | tac
find . ! -path '*bar*' -print
find / -newerct '1 minute ago' -print
find . -mnewer poop
head -c 100 file
tac filename | awk '{for (i=NF; i>1; i--) printf("%s ",$i); printf("%s\n",$1)}'
echo "a,b,c" | tr '\n' ',' | tac -s "," | sed 's/,$/\n/'
su apache -s /bin/ksh
find . -name "*.pl" -exec ls -ld {} \;
find . -name "*.pl" -exec ls -ld {} \;
find . -name core -ok rm {} \;
find /usr -type b -name backup -print
find /usr -type c -name backup -print
find . -atime 7 -print
find .  -ctime +7 -print
find / -links -2 -print
find /  -links 2 -print
find ./ -type f -exec grep https://www.ksknet.net {} \;
find . -regex ".*/my.*p.$" -a -not -regex ".*test.*"
find . -type f -name "*.jpg"
find . -type f -name "*.jpg" -print0 | xargs -0 rename "s/Image_200x200_(\d{3})/img/"
find . \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -ls | awk '{total += $7} END {print total}'
find /home/you -iname "*.mp3" -daystart -type f -mtime 1
find /usr -follow -name '*.sh'
files=`find .`
find . -name test -prune -regex ".*/my.*p.$"
find . -name 'a(b*' -print
find . -regex ".*/my.*p.$"
find . -name test -prune -o -regex ".*/my.*p.$"
find /myfiles -name '*blue*'
grep ^malloc `find src/ -name '*.[ch]'`
find -name '*.undo' -exec wc -c {} + | tail -n 1
find -name '*.undo' -exec wc -c {} + | tail -n 1 | cut -d' ' -f 1
find / -name whatever -not -path "/10_Recommended*" -not -path "/export/repo/*"
find . -name "$pattern"
find /home/ -type f -newer /tmp/after -not -newer /tmp/before
find ./ -regex "./cmn-.\.flac"
find .cache/bower/ -name "message.txt" | xargs cat
find . -name \*.py | xargs grep some_function
find . -name “*.[php|PHP]” -print | xargs grep -HnT “specified string”
find / -name "*.log"
find / -xdev -name "*.log"
find ! -path "dir1" -iname "*.mp3"
find ! -path "dir1" ! -path "dir2" -iname "*.mp3"
find . -type d -name "cpp" -exec find {} -type f \;
find -name file -print
find -name file -quit
find ~/junk  -name 'cart1' -exec mv {} ~/junk/A \;
find . -name '*.py' | xargs grep some_function
find / -path excluded_path -prune -o -type f -name myfile -print
find /home/weedly -name myfile -type f -print
find . -name onlyme.sh -exec pwd \;
find . -name onlyme.sh -execdir pwd \;
find . -name '*.c' | xargs grep 'stdlib.h'
find /directory/containing/files -type f -print0 | xargs -0 grep "test to search"
find . -name "*.c" -exec grep -ir "keyword" {} ";"
find . -type f \( -iname “*.c” \) |grep -i -r “keyword”
find . -type f -exec grep some_string {} \;
find . -exec grep chrome {} +
find . -exec grep chrome {} \;
find . -type f -exec grep 'needle' {} \;
find . | xargs grep 'chrome'
find . -type f -exec grep 'needle' {} \;
find . -exec grep -l foo {} +
find . -type f -exec grep -l 'needle' {} \;
find / -type f -exec grep -Hi 'the brown dog' {} +
find -name '*.[ch]' | xargs grep -E 'expr'
find . -type f | xargs grep "text"
find . -name \*.php -type f -exec grep -Hn '$test' {} \+ | wc -l
find . -name \*.php -type f -exec grep -Hn '$test' {} \; | wc -l
find . -name \*.php -type f -print0 | xargs -0 -n1 grep -Hn '$test' | wc -l
find . -name \*.php -type f -exec grep -Hn '$test' {} \;
find . -name \*.php -type f -print0 | xargs -0 -n1 grep -Hn '$test'
find . -name \*.php -type f -exec grep -Hn '$test' {} \+
find -maxdepth 1 -type f | xargs grep -F 'example'
find -type f -print0 | xargs -r0 grep -F 'example'
find . -type f -exec grep -l linux {} \;
find / -type f -exec grep -i 'the brown dog' {} +;
find . -type f -name "*.sh" -exec grep -l landoflinux {} \;
find . -type f -name "*.sh" -exec grep -il landoflinux {} \;
find . -name "*.cpp" -exec dirname {} + | sort -u
find . -name "*.cpp" -exec dirname {} \; | sort -u
find . -name '*.cpp' | sed -e 's/\/[^/]*$//' | sort | uniq
find . -name '*.tif ' -print
find / -name '*.tif ' –print
find . -type f -name "*.scala" -exec grep -B5 -A10 'null' {} \;
find /home -type d -empty
find /volume1/uploads -name "*.mkv"
find /home/you -iname "*.mp3" -atime 10 -type -f
find . -type f -maxdepth 1 -not -empty -print0 | xargs -0i cp /dev/null {}
find /home/you -iname "*.pdf" -atime -60 -type -f
find /nas/projects/mgmt/scripts/perl -mtime 1 -daystart -iname "*.pl"
find . -type f \( -iname "*.c" -or -iname "*.asm" \)
find . -type f -print0 | xargs -0 grep -cH '' | awk -F: '$2==16'
find . -type f -print0 | xargs -0 grep -cH '.' | grep ':16$'
find . -type f -print | xargs -L1 wc -l
find . -type f -print0 | xargs -0L1 wc -l
find -name *.tar.gz
find /home -name *.txt
find /home/david -mmin -10 -name '*.c'
find . -name *.o -perm 664 -print
find . -name \*.c -print
find / -iname "*.mp3" -type d -exec /bin/mv {} /mnt/mp3 \;
find . -mindepth 1 -maxdepth 1 -type d
find . -size +10k -exec ls -lS {} \+ | head -1
find /store/01 -name "*.fits"
founddata=`find . -name "filename including space" -print0`
find . | awk '{printf "%s ", $0}'
find . | paste -sd " "
find . -name "file2015-0*" -exec mv {} .. \;
find . -name "file2015-0*" | head -400 | xargs -I filename mv  filename
find . -user xuser1 -exec chown -R user2 {} \;
find . -mtime -1
find . -size +10000c -size -32000c -print
find . -group staff -perm -2000 -print
find . \( -name a.out -o -name '*.o' -o -name 'core' \) -exec rm {} \;
find -type d -printf '%d\t%P\n' | sort -r -nk1 | cut -f2-
find . -size +10k -exec ls -ls {} \+ | sort -n | tail -1
find . -name "filename including space" -print0 | xargs -0 rm -rdf
find . -name 'my*'
var="$(find . -name 'gen*.bt2')"
find / \( -perm 2000 -o -perm 4000 \) -print | diff - files.secure
find /data/images -newer /tmp/foo
find /data/images -type f -newer /tmp/start -not -newer /tmp/end
find /home -perm 1553
find /home/user1 -name "*.bin"
find /usr/bin -type f -atime +100
find . -mtime +180 -exec du -sh {} \;
find . -mtime +180 -exec du -ks {} \; | cut -f1 | awk '{total=total+$1}END{print total/1024}'
find ~/tmp -type f -mtime 0 -exec du -ks {} \; | cut -f1 | awk '{total=total+$1}END{print total/1024}'
find -type f -name dummy
find . -name foo.txt -print0 | xargs -0  -I{} mv {} /some/new/location/{}
find /mnt/hda1/zdjecia/test1/ -iname “*.jpg” -type f -exec cp {} -rv /mnt/hda1/test{} ‘;’
find /apps -xdev -name "*.log" -type f -mtime +60 | xargs rm
find / -iname "*.mp3" -exec mv {} /mnt/mp3 \;
find /home/you -iname “*.mp3” -atime 01 -type -f
find . -maxdepth 1 -name '[!.]*' -printf 'Name: %16f Size: %6s\n'
find /nas/projects/mgmt/scripts/perl -mtime 8 -mtime -10 -daystart -iname "*.pl"
find /home/mywebsite -type f -name "*.php" -ctime -30
find . -type f -exec cat {} \;
find /home -type f -perm 0777 -print
find /my/source/directory -ctime -2 -type f -printf "%P\n" | xargs -IFILE rsync -avR /my/./source/directory/FILE /my/dest/directory/
find /etc/ -type f -mtime -1
find . -name 'my*' -type f
find /var/www -type f -name «access.log*» -size +100M
find / -iname "*.mp3" -type f -exec /bin/mv {} /mnt/mp3 \;
find / -iname "*.mp3" -type f -print0 | xargs -0 -I '{}' /bin/mv "{}" /mnt/mp3/
find / -iname "*.mp3" -type f | xargs -I '{}' mv {} /mnt/mp3
find / -xdev -name "*.rpm"
find . -name "*.txt" -execdir ls -la {} ";"
find /foo/ -name "*.txt" -exec rm -v {} \;
find . -name "*.xml" -exec echo {} \;
find . -print|grep ?i dbmspool.sql
find ./test -type d -name '[0-9][0-9][0-9][0-9][0-9]'
find ./test -regextype posix-egrep -type d -regex '.*/[0-9]{5}$'
find ~/junk  -name 'cart[4-6]' -exec rm {}  \;
find / -iname '*python*'
find / -name '*python*'
find . -name "S1A*1S*SAFE"
find ./ -regex '.*\..*'
find -type f -name '*.au'
find . -name '[mM][yY][fF][iI][lL][eE]*'
find . -iname 'MyFile*'
find . -iname "WSFY321.c"
find /work -name 'memo*' -user ann -print
find . -perm -444 -perm /222 ! -perm /111
find . -perm -a+r -perm /a+w ! -perm /a+x
find . -perm -220
find . -perm -g+w,u+w
find . -perm /220
find . -perm /u+w,g+w
find . -perm /u=w,g=w
find /usr -name temp -print
find /mp3collection -name '*.mp3' -size -5000k
find . -name photoA.jpg photoB.jpg photoC.jpg
find ./ -type f -name "pattern" ! -path "excluded path" ! -path "excluded path"
find /users/tom -name "*.pl"
find -name '*.php' -exec grep -iq "fincken" {} \; -exec grep -iq "TODO" {} \; -print
find ./ -type f -name "*" ! -path "./.*" ! -path "./*/.*"
find / -name myfile -type f -print
find . -type l -exec readlink -f '{}' \; | grep -v "^`readlink -f ${PWD}`"
find /tmp -name '*.swp' -exec rm {} \;
find . -type f \( -iname "*.txt" ! -perm -o=w \)
find . -type f \( -iname "*.txt" -not -perm -o=w \)
find . -type f \( -iname "*.txt" -and -perm -o=w \)
find /home/user1 -name '*.txt' | xargs cp -av --target-directory=/home/backup/ --parents
find /home/you -iname "*.txt" -mtime -60 -exec cat {} \;
find /home/hobbes/ /home/calvin/ -name “*.txt”
find . -name config -type d
cd `find . -name "config"`
cd $(find . -name config -type d | sed 1q)
find /etc -name mysql -type d
find / -name mysql -type d
find / -type d -name "ora10"
find . -type d -name 'uploads' -print0 | xargs -0 chmod -R 755
find / -type d -name "ora10*"
find /home -type d -name testdir
find . -type d -name aa
find /nfs/office -name .user.log -print
find . -name abc -or -type d
find -name file -fprint file
find . -path ./.git  -prune -o -name file  -print
find / -name file1
find /path -name file_name
find / -name filename
find / -name foo.txt
find /home/mywebsite -type f -name "foobar.txt"
find /data/SpoolIn -name job.history | xargs grep -o -m 1 -h 'FAIL\|ABOR' | sort | uniq -c
find . -name myfile |& grep -v 'Permission denied'
find ~ -name myletter.doc -print
find ~ -name "name_to_find"
find . -name "process.txt"
find . -iname 'process.txt' -print
find / -name "process.txt"
find / -iname 'process.txt' -print
find /usr -iname centos
find /work -name chapter1
find / -name filename
find . -name foo | xargs ls -tl
find -name foo.txt -execdir vim '{}' ';'
find / -name foo.txt
find / -name ”*filename*”
find /data/SpoolIn -name job.history
find /home/calvin/ -iname “picasso”
find . -name test
find /home /opt -name test.txt
find -name test2
find . -name filename.txt
find . -iname filename.txt
find /home -name filename.txt
find . -name foo.txt
find -name foo.txt -execdir rename 's/\.txt$/.xml/' '{}' ';'
find . -name 'kt[0-9] '
find . -path ./proc -prune -or -path ./sys -prune -or -path ./run -prune  -or -iname '*start*' -print
find /usr -name temp -atime +7 -print
find "Test Folder" -type d -name '.dummy' -delete
find "Test Folder" -type d -name .dummy -exec rm -rf \"{}\" \;
find -depth "Test Folder" -type d -name .dummy -exec rm -rf \{\} \;
find -name “*.xml” -exec grep -l “slc02oxm.us.oracle.com” {} \;
find /etc/ -iname "*" | xargs grep '192.168.1.5'
find . -type f -name \* | grep tgt/etc/file1 tgt/etc/file2 tgt/etc/file3
find / -type f -print0 | xargs -0 grep -i pattern
find / -type f -iname "Dateiname"
find /home -name foo.bar -type f -exec rm -f "{}" ';'
find .  \( -name work -o -name home \)  -prune -o -name myfile -type f -print
find /root/ -name 'work' -prune -o -name myfile -type f -print
find /root/ -path '/root/work' -prune -o -name myfile -type f -print
find . -type f -iname ‘HSTD*’ -daystart -mtime 1 -exec cp {} /path/to new/dir/ \;
find /etc -type f | xargs grep -l -i "damian"
find /path/to/dir -type f -print0 | xargs -0 grep -l "foo"
find /path/to/dir -type f | xargs grep -l "foo"
find /path/to/dir/ -type f -name "file-pattern" -print0 | xargs -I {} -0 grep -l "foo" "{}"
find /mycool/project/ -type f -name "*.py" -print0 | xargs -I {} -0 grep -H --color "methodNameHere" "{}"
find . -iname "*notes*" | xargs grep -i mysql
find . -iname "*notes*" -print0 | xargs -I{} -0 grep -i mysql "{}"
find /etc/ -type f -name "*.conf" -print0 | xargs -I {} -0 grep "nameserver" "{}"
find /etc/ -iname "*" -type f -print0 | xargs -0 grep -H "nameserver"
find /book -print | xargs grep '[Nn] utshell'
find . -name '*bills*' -exec grep -H "put" {} \;
find . -type f -exec grep -i “redeem reward” {} \; -print
find . -type f | xargs grep -l "search-pattern"
find ./ -exec grep -q 'slrn' '{}' \; -print
find $HOME/html/andrews-corner -exec grep -q 'slrn' '{}' \; -print
find ~jsmith -exec grep LOG '{}' /dev/null \; -print
find / -type f -exec grep bananas {} \; -print
find . -name "*.bash" |xargs grep "echo"
find . -name "*.xml" -exec grep "ERROR" /dev/null '{}' \+
find . -name "*.js" -exec grep -iH foo {} \;
grep -iH foo `find . -name "*.js"`
find /tmp -type f -name ‘*.txt*’ | sed -e ‘s/.*/\”&\”/’ |xargs -n 1 grep -l hello|sed -e ‘s/.*/\”&\”/’|xargs -n 1 rm -f
find . -iname '*py' -exec grep "text" {} \;
find /home/*/public_html/ -type f -iwholename "*/modules/system/system.info" -exec grep -H "version = \"" {} \;
find ~/mail -type f | xargs grep "Linux"
find . -type f -exec grep -l "word" {} +
find .  -size -10c -print
find . -name Chapter1 -type f
find . -type d -name test
find . -type f -name test
find . -name myletter.doc -print
find . -name test
find . -iname test
find . -name 'myletter*' -print
find /home/ -name monfichier
find /usr -type f -name backup -print
find / -name jan92.rpt -print
find "$HOME/" -name myfile.txt -print
find . -name myfile.txt -print
find Lib/ -name '*.c' -print0 | xargs -0 grep ^PyErr
find . -type f -print0 | xargs -0 -e grep -nH -e MySearchStr
find ./ -type f -exec sed -i '' 's#NEEDLE#REPLACEMENT#' *.php {} \;
find /usr /home -name Chapter1.txt -type f
find /usr -name "Chapter*" -type f
find /usr/local -name "*.html" -type f
find / -name Chapter1 -type f
find / -name Chapter1 -type f -print
find / -name Chapter1 -type f
find / -name Chapter1 -type f -print
find / -name *.mp3
myVariable=$(env  | grep VARIABLE_NAME | grep -oe '[^=]*$');
echo "* * * * * touch $(pwd)/washere1" | crontab
echo "30 * * * * touch $(pwd)/washere2" | crontab
uname_m=`uname -m`
r="$(uname -r)"
find . -maxdepth 1 -name \*.txt -print0 | grep -cz .
find ~/src -type f \( -iname '*.cpp' -or -iname '*.h' -or -iname '*.c' -or -iname '*.hpp' \) -exec echo {} \;
find . -type d \( -name media -o -name images -o -name backups \) -prune -o -print
find . -path './media' -prune -o -path './images' -prune -o -path './backups' -prune -o -print
find /usr/tom | egrep '*.pl| *.pm'
find .
find . -print
find * -prune -type f -size +0c -print
find /home -type f -name '*.aac'
find . -maxdepth 1 -type f -name '*.flac'
find . -type d
find .
find . -type d -name '.git*' -prune -o -type f -print
find . -name ‘*ITM*’
find / -size +1.1G
find / -size +100M
find /etc -mtime -1
find /home -type f -name '*.mp3'
find . -type f -print0
find /home/user/demo -type f -print
find . -type f -size +50000k | xargs du -sh
find / -type f -size +100M | xargs du -sh
find . -mtime 0 -print
find / -user test1 -exec du -sm {} \;|awk '{s+=$1}END{print s}'
su -l builder
su -
sleep 1
sleep 10
sleep 5
sleep 500
find .  -name "*.txt" -type f -daystart -mtime -91 -mtime +2 | xargs cat | sort | uniq
find / -type f -printf "\n%AD %AT %p" | head -n 11 | sort -k1.8n -k1.1nr -k1
find . -type f -exec ls -s {} \; | sort -n -r | head -10
find . -type f -exec ls -s {} \; | sort -n | head -10
find bills -type f -execdir sort -o '{}.sorted' '{}' ';'
find bills -type f | xargs -I XX sort -o XX.sorted XX
split -l 100 "$SOURCE_FILE"
split -l 600 list.txt
find /dev/shm/split/ -type f -exec split -l 1000 {} {} \;
find posns -type f -exec split -l 10000 {} \;
tar --one-file-system -czv /home | split -b 4000m - /media/DRIVENAME/BACKUPNAME.tgz
tar czf - www|split -b 1073741824 - www-backup.tar.
cat file1 file2 ... file40000 | split -n r/1445 -d - outputprefix
tail -n +2 file.txt | split -l 4 - split_
cat inputfile | grep "^t\:" | split -l 200
cat *.txt | tail -n +1001 | split --lines=1000
split -a 5 $file
split --number=l/6 ${fspec} xyzzy.
split -l9 your_file
split -n 1000000 /etc/gconf/schemas/gnome-terminal.schemas
split -n 1000 /usr/bin/firefox
split -n 100000 /usr/bin/gcc
split --bytes=1500000000 abc.txt abc
split -l 100 date.csv
split --lines=1 --suffix-length=5 input.txt output.
split --lines=30000000 --numeric-suffixes --suffix-length=2 t.txt t
ls | split -l 500 - outputXYZ.
sed 's/\(.....\)\(.....\)/\1\n\2/' input_file | split -l 2000000 - out-prefix-
sed 's/3d3d/\n&/2g' temp | split -dl1 - temp
tar [your params] |split -b 500m - output_prefix
split --lines=50000 /path/to/large/file /path/to/output/file/prefix
split -n2 infile
split -C 100m -d data.tsv data.tsv.
split -a4 -d -l100000 hugefile.txt part.
sed 100q datafile | split -C 1700 -
ping -c 25 google.com | tee >(split -d -b 100000 - /home/user/myLogFile.log)
ssh -l buck hostname
ssh buck@hostname
ssh myusername@ssh.myhost.net "mkdir -p $2"
ssh vagrant@127.0.0.1 -p 2222 -o Compression=yes -o DSAAuthentication=yes -o LogLevel=FATAL -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -i ~/.vagrant.d/less_insecure_private_key -o ForwardAgent=yes
ssh -p 10022 localhost
ssh -p 4444 localhost
sleep 100 | sleep 200 &
find . -wholename './src/emacs' -prune -o -print
find . -wholename './src/emacs' -prune , -print
find . -wholename './src/emacs' -prune -print -o -print
su username
find . -ok tar rvf backup {} \;
find . -type f -name "*.java" | xargs tar cvf myfile.tar
diff --brief <(awk '{print $2}' A) <(tac B | awk '{print $2}')
find . -type f -iname '*.mp3' -exec rename '/ /_/'  {} \;
find / -name '#*' -atime +7 -print | xargs rm
find . -iname '*.jar' -printf "unzip -c %p | grep -q '<stringWithOrWithoutSpacesToFind>' && echo %p\n" | sh
find /u/netinst -print | xargs chmod 500
find . -name '*.php' -exec chmod 755 {} \; | tee logfile.txt
find folder_name -type d -exec chmod 775 ‘{}’ \;
find . -name "*.txt" -exec echo {} \; -exec grep banana {} \;
find . -name "*.txt" \( -exec echo {} \; -o -exec true \; \) -exec grep banana {} \;
find ./ -type f \( -iname \*.jpg -o -iname \*.png \)
find . * | grep -P "[a-f0-9\-]{36}\.jpg"
find . -regextype posix-egrep -regex '\./[a-f0-9\-]{36}\.jpg'
find . -regextype sed -regex ".*/[a-f0-9\-]\{36\}\.jpg"
find . ... -exec cat {} \; -exec echo \;
mkdir -pv /tmp/boostinst
man find
echo "hello `sleep 2 &`"
bind -m vi-insert '"{" "\C-v{}\ei"'
