package com.resumeBuilder.view.user;

import com.resumeBuilder.controller.user.UserLoginListener;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;

public class UserLoginView extends JFrame {

    private JPanel contentPane;
    private JTextField txtUsrName;
    private JPasswordField txtPwd;
    private JLabel lblUsrName;
    private JLabel lblPassWord;
    private JButton btnUsrLogin;
    private JLabel lblTitle;
    private JButton btnBack;
    private JLabel lblBack;

    public UserLoginView() {
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setBounds(100, 100, 450, 300);
        contentPane = new JPanel();
        contentPane.setBackground(Color.LIGHT_GRAY);
        contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
        setContentPane(contentPane);
        contentPane.setLayout(null);

        lblUsrName = new JLabel("User name  ");
        lblUsrName.setForeground(Color.BLACK);
        lblUsrName.setBackground(Color.BLACK);
        lblUsrName.setFont(new Font("Tahoma", Font.PLAIN, 14));
        lblUsrName.setBounds(65, 103, 93, 23);
        contentPane.add(lblUsrName);

        lblPassWord = new JLabel("Password");
        lblPassWord.setForeground(Color.BLACK);
        lblPassWord.setFont(new Font("Tahoma", Font.PLAIN, 14));
        lblPassWord.setBounds(65, 144, 68, 23);
        contentPane.add(lblPassWord);

        txtUsrName = new JTextField();
        txtUsrName.setBounds(158, 106, 157, 20);
        contentPane.add(txtUsrName);
        txtUsrName.setColumns(10);

        btnUsrLogin = new JButton("Login");
        btnUsrLogin.addActionListener(new UserLoginListener(this));

        btnUsrLogin.setBackground(Color.WHITE);
        btnUsrLogin.setFont(new Font("Tahoma", Font.BOLD, 14));
        btnUsrLogin.setBounds(173, 211, 93, 23);
        contentPane.add(btnUsrLogin);

        txtPwd = new JPasswordField();
        txtPwd.setBounds(158, 144, 157, 20);
        contentPane.add(txtPwd);

        lblTitle = new JLabel("User Login");
        lblTitle.setFont(new Font("Tahoma", Font.BOLD, 14));
        lblTitle.setBounds(173, 28, 80, 23);
        contentPane.add(lblTitle);

        btnBack = new JButton("<--");
        btnBack.addActionListener(new UserLoginListener(this));
        btnBack.setFont(new Font("Tahoma", Font.BOLD, 16));
        btnBack.setBounds(10, 11, 75, 23);
        contentPane.add(btnBack);

        lblBack = new JLabel("Back");
        lblBack.setHorizontalAlignment(SwingConstants.CENTER);
        lblBack.setBounds(10, 34, 75, 14);
        contentPane.add(lblBack);
    }

    public String getTxtUsrName() {
        return txtUsrName.getText();
    }

    public String getPasswordField() {
        return txtPwd.getText();
    }
}