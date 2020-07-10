// import java.rmi.*;
// import java.rmi.registry.*;
// import java.rmi.server.*;
// import java.net.*;
import java.io.*;

public class Sample{
    public static void main(String[] args) {
        BufferedReader userInput = new BufferedReader(new InputStreamReader(System.in));
        String cmd;
        String out;
        try{
            while(true){
                cmd = userInput.readLine();
                out = "";
                //Process proc = Runtime.getRuntime().exec(new String[] {"sh", "-c", "zip -r buffer zip \"/home/hitarth/Downloads/Wallpapers\""}, null, null);
                Process proc = Runtime.getRuntime().exec(new String[] {"bash", "-c", cmd}, null, null);
                if(proc.waitFor() != 0){
                    System.out.println("cmd 1 failed...");
                }


                BufferedReader stdInput = new BufferedReader(new InputStreamReader(proc.getInputStream()));
                String s = null;
                while ((s = stdInput.readLine()) != null) {
                    out += s;
                    out += "\n";
                }            
                stdInput.close();
                proc.destroy();
                System.out.println(out);
            }
        }
        catch(Exception e){
            e.printStackTrace();
        }
    }
}