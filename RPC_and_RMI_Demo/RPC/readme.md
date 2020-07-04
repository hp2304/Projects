# Fun experiment

* This is a demo program for **RPC (Remote Procedure Call)**. RPC (*kinda middleware*) is developed by *SUN Microsystems* to build distributed systems. I highly recommend (*well require*) to read slides 1 to 38 in **Fundamentals of Distributed Computing.pdf**, it will explain the the need for RPC, abstract design and working of RPC, etc. Read [this](https://docs.oracle.com/cd/E19253-01/816-1435/rpcgenpguide-24243/index.html) too. Only after understanding these stuff, move ahead.



* It's like an API call. Server has some procedures (*functions*) and it's listening for client requests. Client need to know about *ip address* of the server to call any function (implemented by the server) remotely on the server.

* Here I have implemented just a function for sorting an array. For demostration purposes, I have used *insertion sort* algo at the server side. So client program will read an array, call the sort function remotely, get response from server and display the results. This demo program implementation is in *C lang*.

---
#### Note
* Here for this demo, client and server programs need not be running on same machine, but they have to be in **same network**.
---

## Requirements
* *rpcgen*. Usually it's installed in linux systems. If not *sudo apt-get install rpcgen*.

## Usage
```bash
foo@bar:~$ rpcgen sort_arr.x
```
It will generate these 4 files, **sort_arr.h** (*header to be included in both server and client programs*), **sort_arr_clnt.c** (*client stub program*), **sort_arr_svc.c** (*server stub program*) and **sort_arr_xdr.c** (for *external data representation*).

Copy sort_arr.h, sort_arr_clnt.c and sort_arr_xdr.c to *client* machine along with client.c.

Copy sort_arr.h, sort_arr_svc.c and sort_arr_xdr.c to *server* machine along with server.c.

No need to copy if you are running both on same machine and if all these files are in same directory.

First run server as a background process (By appending *&* at the end).
```bash
foo@bar:~$ cc server.c sort_arr_svc.c sort_arr_xdr.c -o serv
foo@bar:~$ ./serv &
```
Now our server is on and listening for client requests.

To run client program, you will need *ip address* of server machine (use *ifconfig* at server side to find out. Can use localhost, if both are on same machine). Give a list of space seprated numbers in args, which we want to sort. You can give arbitary number of numbers as argument :)
```bash
foo@bar:~$ cc client.c sort_arr_clnt.c sort_arr_xdr.c -o cli
foo@bar:~$ ./cli localhost 231 34 5 4 23 54 76 343 76 343 121 345 435 5747 32 22 443 687
```
Client program will **call the sort procedure remotely on server** (*just like API call*). Upon receival of the request, server will sort the input arr and will return the sorted array as response to the client.

At client side,
```bash
foo@bar1:~$ ./cli localhost 21 32 11
3
21 32 11 

Got response from server
3
11 21 32
```

```bash
foo@bar3:~$ ./cli localhost 231 34 5 4 23 54 76 343 76 343 121 345 435 5747 32 22 443 687
18
231 34 5 4 23 54 76 343 76 343 121 345 435 5747 32 22 443 687 

Got response from server
18
4 5 22 23 32 34 54 76 76 121 231 343 343 345 435 443 687 5747 
```

On server side,
```bash
foo@bar2:~$ ./serv



Got client request...
Got array of size 3.
21 32 11 
Performing sorting algorithm...
Returning sorted array...


Got client request...
Got array of size 18.
231 34 5 4 23 54 76 343 76 343 121 345 435 5747 32 22 443 687 
Performing sorting algorithm...
Returning sorted array...
```



* Add other **complex procedures** in this and experiment :)
