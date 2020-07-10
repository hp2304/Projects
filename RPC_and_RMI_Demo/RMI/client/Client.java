//package client;
/* Client
 * A client who communicates with the
 * Server with RMI.
 */

import java.rmi.*;
import java.net.*;
import java.io.*;
import java.util.Random;

public class Client {

    MyInterface server;

    public Client(String host, int port) throws RemoteException, NotBoundException, MalformedURLException{
        System.out.println(port);
        String name = "//" + host + ":" + port + "/Server";
        System.out.println(name);
        server = (MyInterface) Naming.lookup(name);
    }

    public void askServer() {
        boolean keepGoing = true;
        BufferedReader userInput = new BufferedReader(new InputStreamReader(System.in));
        String request;

        // String with command information
        String commands = "Enter linux command to get its output (Enter exit to quit) :\n";
        System.out.println(commands);

        while (keepGoing) {
            System.out.print("\n>> ");
            try{
                request = userInput.readLine();
                System.out.println();
                String words[] = request.split(" ");
                if(request.equals("exit"))
                    keepGoing = false;
                else if(words[0].equals("get")){
                    words = request.split(" ", 2);
                    byte[] filedata = server.downloadFile(words[1].trim());
                    if(filedata==null){
                        System.out.println("Byte array is null, file doesn't exist at the location...");
                        continue;
                    }
                    String fname = "my_drive_" + Integer.toString(new Random(System.currentTimeMillis()).nextInt(1000)) + ".zip";
                    File file = new File(fname);
                    BufferedOutputStream output = new BufferedOutputStream(new FileOutputStream(file.getName()));
                    output.write(filedata,0,filedata.length);
                    output.flush();
                    output.close();
                    System.out.println("Received "+fname+" in current dir.");
                }
                else{
                    System.out.print(server.sendCmdOutput(request));
                }
            } catch (RemoteException re) {
                re.printStackTrace();
            } catch (IOException ioe) {
                System.out.println("Could not read from standard in.");
            }
        }
    }

    public static void main(String args[]) {
        String host;

        if (args.length > 0) 
            host = args[0];
        else
            host = "localhost";

        try {
            Client wc = new Client(host, Integer.parseInt(args[1]));
            wc.askServer();
        } catch(RemoteException re) {
            System.out.println("Remote exception");
            re.printStackTrace();
        } catch(MalformedURLException mue) {
            System.out.println("The host " + host + " was malformed.");
        } catch(NotBoundException nbe) {
            System.out.println("The name was not bound.");
        }
    }
}

