## Demo Program For Remote Method Invocation (RMI)

* This is a demo program for **RMI (Remote Method Invocation)**. RMI (*kinda middleware*) was developed to build distributed systems in *JAVA*. 

* Here I have implemented a client-server program using which one can remotely execute *shell* commands (kinda *remote shell*) in remote machine. And can also download file/directory from remote machine.

### Requirements
* Understand **RPC** demo program first.
* Read slides 39 to 80 in **Fundamentals of Distributed Computing.pdf**.
* *Java* has to be installed in both client and server machines.

---
#### Note
* Here for this demo, client and server programs need not be running on same machine, but they have to be in **same network**.
---

## Usage
Copy *client/* directory in client machine and copy *server/* directory in server machine.

In server machine,
```bash
foo@server:~$ rmiregistry 8090 &
foo@server:~$ javac Server.java MyInterface.java
foo@server:~$ rmic -verbose Server
```
Rmi registry acts as kind of *DNS*, which registers various servers'. Last command will create *Server_Stub.class* file, copy this file to *client/* directory in client machine. Now run server program,
```bash
foo@server:~$ java Server &
```
Now our server is running on port 4711 (You can change this if you want, change line 123 and 124 *Server.java*). 

In client machine,
```bash
foo@client:~$ javac Client.java MyInterface.java
foo@client:~$ java Client <server ip address> <server port: 4711>
```
To run client program server's ip address and port (one on which server program is running, 4711 in this case) is required. Here *MyInterface.java* is same file as in *server/* directory.

Upon successful execution, client program will display terminal like interface, in which you can run bash commands as in local terminal. You can execute *cd* commands too, can navigate anywhere in server machine. Now to download a file/directory from server, run this command in our client program, 

```bash
> get {file/directory name}
```
This command will download a zip file (Containing our desired file/directory) in directory, where in *Client* executable is running (*client/* directory in this case).

---
#### Note

* To execute *sudo* commands, server machine's password is required obviously, run below command in our client program to execute sudo command (because directly executing it will prompt for *password* as input),

```bash
> echo <server password> | sudo -S <command to be executed>
```
* Here our shell interface is *not color coded* like in local terminal and there will be *no suggestions (*auto completion*)* for commands as in local terminal. You have to write whole commands yourself and execute it.

---

* This all can be done by writing single line using *ssh* and *scp* commands. And it's secure too. Then why do all this? because it's something implemented by you (not RMI though :)  ). It's fun :)