/*
Copyright 2018 XIAOLIN WANG (xiaolin.wang@nict.go.jp; arthur.xlw@gmail.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef _CYTONLIB_XLCLIB_H_
#define _CYTONLIB_XLCLIB_H_


#include <iostream>
#include <fstream>
#include <ostream>
#include <stdarg.h>
#include <vector>
#include <assert.h>
#include <map>
#include <sstream>
#include <list>
#include <limits>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <dirent.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include  <stdio.h>
#include  <stdlib.h>
#include <errno.h>

#include <tr1/unordered_map>
#include <tr1/unordered_set>

#include <math.h>
#include <cmath>
#include <algorithm>
#include <numeric>

class XLLibTime
{
public:
	clock_t clock;
	double time;
};


class XLLib
{
public:
static void end(XLLibTime& xlTime) {

		printf("OK.");

		if (xlTime.clock >= 0) {

			double time = (double)(clock() - xlTime.clock)/CLOCKS_PER_SEC;

			struct timeval utime;
			gettimeofday(&utime, NULL);
			double time_cur_user = double(utime.tv_sec) + 1e-6 * utime.tv_usec;
			double time_user=time_cur_user-xlTime.time;

			double diff=fabs(time-time_user);
			if(diff>1e-3){
				printf(" (%.3f/%.3f secs)", time_user, time);
			}else{
				printf(" (%.3f secs)", time);
			}
		}

		printf("\n");
	}

static void writeFile(const std::string & fileName,	const std::vector<std::string> & lines)
	{
		XLLib::dirPrepare4file(fileName);
		std::ofstream f(fileName.c_str());
		for (unsigned int i = 0; i < lines.size(); i++)
		{
			f << lines[i];
			f << "\n";
		}
		f.close();
	}

template<typename T> static std::string toString_split(const std::vector<T>& ds,
		const std::string& split) {
	std::ostringstream os;
	bool first=true;
	for (typename std::vector<T>::const_iterator d = ds.begin();
			d != ds.end(); d++) {
		if(!first){
			os<<split;
		}
		os<<*d;
		first=false;
	}
	std::string t=os.str();
	return t;
}


template<typename T> static std::string toString_vec(const typename std::vector<T>::const_iterator& iBegin,
		const typename std::vector<T>::const_iterator& iEnd, std::string fmt, const std::string& split = " ") {
	std::string t;
	for (typename std::vector<T>::const_iterator d = iBegin;
			d != iEnd; d++) {
		t += stringFormat(fmt, *d);
		t += split;
	}
	if (t.size() > 0) {
		t.erase(t.size() - 1, 1);
	}
	return t;
}

template<typename T> static std::string toString_vec(const std::vector<T>& ds,
		std::string fmt, const std::string& split = " ") {
	return toString_vec<T>(ds.begin(),ds.end(),fmt,split);
}

	static void fileReader(const std::string& fileName, std::ifstream& f){
		f.open(fileName.c_str());

		if (!f.good()) {
			fprintf(stderr, "cannot find file: %s!\n", fileName.c_str());
			assert(false);
			exit(0);
		}
	}

	static void str2doubles(const std::string& t, std::vector<double>& ds) {
		std::vector<std::string> items;
		str2list(t, items);
		for (unsigned int i = 0; i < items.size(); i++) {
			double d = atof(items[i].c_str());
			ds.push_back(d);
		}
	}

	static void str2doubles(const std::string& t, const std::string splitor,  std::vector<double>& ds) {
		std::vector<std::string> items;
		str2list(t, splitor, items);
		for (unsigned int i = 0; i < items.size(); i++) {
			double d = atof(items[i].c_str());
			ds.push_back(d);
		}
	}

	static std::string fileName(const std::string& path){
		int i=path.length()-1;
		for(;i>=0;i--){
			char ch=path[i];
			if(ch == '/'){
				break;
			}
		}
		if(i>=0){
			return path.substr(i+1);
		}else{
			return (std::string)path;
		}
	}


static void fileLink(std::string src, std::string des)
	{
		if(XLLib::fileExists(des))
		{
			XLLib::fileRemove(des);
		}

		std::string cmd=XLLib::stringFormat("ln -s %s %s", src.c_str(), des.c_str());
		system(cmd.c_str());
	}

	static int fileRemove(std::string src)
	{
		int res=remove(src.c_str());
		return res;

	}


	static XLLibTime startTime()
	{
		XLLibTime res;
		res.clock = clock();

		//user time
		struct timeval utime;
		gettimeofday(&utime, NULL);
		res.time = double(utime.tv_sec) + 1e-6 * utime.tv_usec;
		return res;
	}


	static std::string endTime(const XLLibTime& startTime, double* pTime=NULL)
	{

		std::ostringstream os;
		double time = (double)(clock() - startTime.clock)/CLOCKS_PER_SEC;

		struct timeval utime;
		gettimeofday(&utime, NULL);
		double timeCurUser = double(utime.tv_sec) + 1e-6 * utime.tv_usec;
		double timeUser=timeCurUser-startTime.time;

		double diff=fabs(time-timeUser);
		double tMax=std::max(time, timeUser);
		if(pTime!=NULL)
		{
			*pTime=tMax;
		}

		if(diff>1e-2*tMax){
			os<<stringFormat("%.3e/%.1es", timeUser, time);
		}else{
			os<<stringFormat("%.3es", time);
		}

		return os.str();
	}


	static bool fileExists(std::string fileName){
		FILE *fp = fopen(fileName.c_str(),"r");
		bool res=false;
		if(fp!=NULL){
			res=true;
			fclose(fp);
		}
		return res;
	}



	static bool fileExists(const char*fileName){
		FILE *fp = fopen(fileName,"r");
		bool res=false;
		if(fp!=NULL){
			res=true;
			fclose(fp);
		}
		return res;
	}

	static std::string stringTrim(const std::string &str) {
		int s = str.size(), k = 0, k2 = s - 1;
		while (k < s && (str[k] == ' ' || str[k] == 0x09 || str[k] == '\n')) {
			k++;
		}

		while (k2 >= 0 && (str[k2] == ' ' || str[k2] == 0x09 || str[k2] == '\n')) {
			k2--;
		}

		if (k2 >= k){
			return str.substr(k, k2 + 1 - k);
		}else{
			return (std::string) "";
		}
	}


	static std::string printfln(const std::string & fmt, ...)
	{
		int size = 100;
		std::string str;
		va_list ap;
		while (1) {
			str.resize(size);
			va_start(ap, fmt);
			int n = vsnprintf((char *) str.c_str(), size, fmt.c_str(), ap);
			va_end(ap);
			if (n > -1 && n < size) {
				str.resize(n);
				break;
			}
			if (n > -1)
				size = n + 1;
			else
				size *= 2;
		}

		std::cout<<str<<"\n";

		fflush(stdout);

		return str;
	}

	static void printfln(std::ostream & os, const std::string & fmt, ...) {

		int size = 100;
		std::string str;
		va_list ap;
		while (1) {
			str.resize(size);
			va_start(ap, fmt);
			int n = vsnprintf((char *) str.c_str(), size, fmt.c_str(), ap);
			va_end(ap);
			if (n > -1 && n < size) {
				str.resize(n);
				break;
			}
			if (n > -1)
				size = n + 1;
			else
				size *= 2;
		}
		os << str << "\n";

		return;
	}

	static void printfln(std::ostream * os, const std::string & fmt, ...) {

		int size = 100;
		std::string str;
		va_list ap;
		while (1) {
			str.resize(size);
			va_start(ap, fmt);
			int n = vsnprintf((char *) str.c_str(), size, fmt.c_str(), ap);
			va_end(ap);
			if (n > -1 && n < size) {
				str.resize(n);
				break;
			}
			if (n > -1)
				size = n + 1;
			else
				size *= 2;
		}
		*os << str << "\n";

		return;
	}


static void printfln(bool outputStd,std::ostream * os, const std::string & fmt, ...) {

		if(!outputStd && os==NULL){
			return;
		}
		int size = 100;
		std::string str;
		va_list ap;
		while (1) {
			str.resize(size);
			va_start(ap, fmt);
			int n = vsnprintf((char *) str.c_str(), size, fmt.c_str(), ap);
			va_end(ap);
			if (n > -1 && n < size) {
				str.resize(n);
				break;
			}
			if (n > -1)
				size = n + 1;
			else
				size *= 2;
		}
		if(outputStd){
			std::cout<<str<<"\n";
		}
		if(os!=NULL){
			*os << str << "\n";
		}

		return;
	}

	static std::string printf(const std::string & fmt, ...) {

		int size = 100;
		std::string str;
		va_list ap;
		while (1) {
			str.resize(size);
			va_start(ap, fmt);
			int n = vsnprintf((char *) str.c_str(), size, fmt.c_str(), ap);
			va_end(ap);
			if (n > -1 && n < size) {
				str.resize(n);
				break;
			}
			if (n > -1)
				size = n + 1;
			else
				size *= 2;
		}

		::printf("%s",str.c_str());
		if (!str.empty()) {
			fflush(stdout);
		}

		return str;
	}

	static void printf(bool outputStd,std::ostream * os, const std::string & fmt, ...) {
		if(!outputStd && os==NULL){
			return;
		}
		int size = 100;
		std::string str;
		va_list ap;
		while (1) {
			str.resize(size);
			va_start(ap, fmt);
			int n = vsnprintf((char *) str.c_str(), size, fmt.c_str(), ap);
			va_end(ap);
			if (n > -1 && n < size) {
				str.resize(n);
				break;
			}
			if (n > -1)
				size = n + 1;
			else
				size *= 2;
		}
		if(outputStd){
			std::cout<<str;
			std::cout.flush();
		}
		if(os!=NULL){
			*os << str;
			os->flush();
		}

		return;
	}


	static std::string stringFormat(const std::string& fmt, ...) {
		int size = 100;
		std::string str;
		va_list ap;
		while (1)
		{
			str.resize(size);
			va_start(ap, fmt);
			int n = vsnprintf((char *) str.c_str(), size, fmt.c_str(), ap);
			va_end(ap);
			if (n > -1 && n < size)
			{
				str.resize(n);
				break;
			}
			if (n > -1)
			{
				size = n + 1;
			}
			else
			{
				size *= 2;
			}
		}
		return str;
	}


	template<typename T> static std::string toString(const std::vector<T>& ds) {
		std::ostringstream os;
		bool first=true;
		for (typename std::vector<T>::const_iterator d = ds.begin();
				d != ds.end(); d++) {
			if(!first){
				os<<" ";
			}
			os<<*d;
			first=false;
		}
		std::string t=os.str();
		return t;
	}

	static bool stringStartsWith(const std::string& a, const std::string& b) {
		bool res = false;
		int lenA=a.size();
		int lenB=b.size();
		if (lenA >= lenB) {
			int x=a.compare(0, lenB, b);
			if (x == 0) {
				res = true;
			}
		}
		return res;
	}

	static std::vector<std::string> readFile(const char*fileName) {

		std::ifstream f(fileName);
		if (!f.good()) {
			fprintf(stderr, "cannot find file: %s!\n", fileName);
			exit(0);
		}

		std::string curline;
		std::vector<std::string> res;
		while (getline(f, curline, '\n')) {
			curline = stringTrim(curline);
			res.push_back(curline);
		}
		return res;
	}


	static void str2list(const std::string& t, std::vector<std::string>& list, bool clearList=true) {
		if(clearList){
			list.clear();
		}
		size_t i0 = 0;
		unsigned int len=t.length();
		while(i0<len){
			size_t i1 = i0;
			for(;i1<len;i1++){
				if(t[i1]!=' '&&t[i1]!='\t'){
					break;
				}
			}
			if(i1==len){
				break;
			}

			size_t i2=i1+1;
			for(;i2<len;i2++){
				if(t[i2]==' '||t[i2]=='\t'){
					break;
				}
			}
			list.push_back(t.substr(i1, i2-i1));
			i0=i2+1;
		}
	}


	static void str2list(const std::string& str, const std::string& splitor, std::vector<std::string>& list,
			bool clearList=true) {
		if(clearList){
			list.clear();
		}

		if(!str.empty()){
			size_t pos = 0;

			size_t last = pos;
			while ((pos = str.find(splitor, last)) != std::string::npos) {
				list.push_back(str.substr(last, pos - last));
				last = pos + splitor.length();
			}
			list.push_back(str.substr(last));
		}
	}

	static void readFile(std::istream& is, 	std::vector<std::string>& lines){
		std::string line;
		while (true){
			std::getline( is, line ); // Read a line from standard input
			if ( !is.good() ){
				break; // <<EOF>> marker is for use with sockets
			}else{
				lines.push_back(line);
			}
		}

	}

	static void readFile(const std::string& fileName,
			std::vector<std::string>& lines) {

		std::ifstream f(fileName.c_str());
		if (!f.good()) {
			fprintf(stderr, "cannot find file: %s!\n", fileName.c_str());
			exit(0);
		}

		std::string curline;
		while (getline(f, curline, '\n')) {
			curline = stringTrim(curline);
			lines.push_back(curline);
		}
	}

	static void dirPrepare4file(const std::string& path) {
		std::string dir=dirName(path);
		if(dir.length()>0){
			dirMake(dir);
		}
	}

	static bool dirExists(const std::string& wd) {
		DIR *pDir;
		bool bExists = false;

		pDir = opendir(wd.c_str());

		if (pDir != NULL) {
			bExists = true;
			(void) closedir(pDir);
		}
		return bExists;
	}


	static void dirMake(const std::string& wd) {
		if (!dirExists(wd)) {
			std::string cmd = stringFormat("mkdir -p %s", wd.c_str());
			system(cmd.c_str());
		}
	}

	static std::string dirName(const std::string& path){
		int i=path.length()-1;
		for(;i>=0;i--){
			char ch=path[i];
			if(ch == '/'){
				break;
			}
		}
		if(i>=0){
			return path.substr(0,i);
		}else{
			return "";
		}
	}


	static void str2ints(const std::string& t, std::vector<int>& ds) {
		std::vector<std::string> items;
		str2list(t, items);
		for (unsigned int i = 0; i < items.size(); i++) {
			int d = atoi(items[i].c_str());
			ds.push_back(d);
		}
	}

	static void str2ints(const std::string& t, const std::string splitor, std::vector<int>& ds) {
		if(t=="none")
		{
			return;
		}
		std::vector<std::string> items;
		str2list(t, splitor, items);
		for (unsigned int i = 0; i < items.size(); i++) {
			int d = atoi(items[i].c_str());
			ds.push_back(d);
		}
	}

	static void stringRemoveLast(std::string& str) {
		if (!str.empty()) {
			str.erase(str.size() - 1);
		}
	}

	template<typename T> 
  static std::string toString_vec_ostream(const std::vector<T>& vec,
			std::string split) {
		std::ostringstream os;
		typename std::vector<T>::const_iterator i;
		bool first=true;
		for (i = vec.begin(); i != vec.end(); i++) {
			if(!first)
			{
				os<< split;
			}
			first=false;
			os << *i;
		}
		std::string t = os.str();
		return t;
	}

	template<class V, class T>
	class SortNode {
	public:
		V val;
		T t;
		SortNode(const V & val, const T & t) :
			val(val), t(t) {
		}

	};

	template<class V, class T>
	static bool sortCompare(SortNode<V, T> & a, SortNode<V, T> & b) {
		return a.val >= b.val;
	}

	template<class V, class T>
	static void sort(const std::vector<V> & vals, std::vector<T> & ts,
			std::vector<T> *pRes = NULL) {
		std::list<SortNode<V, T> > nodes;
		assert(vals.size() == ts.size());
		for (uint i = 0; i < vals.size(); i++) {
			nodes.push_back(SortNode<V, T>(vals[i], ts[i]));
		}

		nodes.sort(sortCompare<V, T>);

		if (pRes == NULL) {
			pRes = &ts;
		}

		(*pRes).clear();
		typename std::list<SortNode<V, T> >::iterator iter;
		for (iter = nodes.begin(); iter != nodes.end(); iter++) {
			(*pRes).push_back(iter->t);
		}
	}

	template<class V>
	static void sortIndex(const std::vector<V>& vals, std::vector<int>& res)
	{
		std::list<SortNode<V, int> > nodes;
		std::vector<int> ts;
		for(uint i=0; i<vals.size(); i++)
		{
			ts.push_back(i);
		}
		sort(vals, ts, &res);

	}


	template<typename T> static int iMax(T* data, int len)
	{
		T res=data[0];
		int idx=0;
		for(int i=1; i<len; i++)
		{
			if(res< data[i])
			{
				res=data[i];
				idx=i;
			}
		}
		return idx;
	}

};




#endif /* XLCLIB_H_ */
