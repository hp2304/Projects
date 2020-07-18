package com.resumeBuilder.view.company;

import com.resumeBuilder.controller.company.CompanyLoginView;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;

public class CompanyLogin extends JFrame {

    private JPanel contentPane;
    private JTextField txtUserName;
    private JPasswordField txtPwd;
    private JLabel lblLogin;
    private JLabel lblUserName;
    private JLabel lblPwd;
    private JButton btnLogin;
    private JButton btnBack;
    private JLabel lblBack;

    public CompanyLogin() {
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setBounds(100, 100, 450, 300);
        contentPane = new JPanel();
        contentPane.setBackground(Color.LIGHT_GRAY);
        contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
        setContentPane(contentPane);
        contentPane.setLayout(null);

        lblLogin = new JLabel("Company Login");
        lblLogin.setHorizontalAlignment(SwingConstants.CENTER);
        lblLogin.setForeground(Color.BLACK);
        lblLogin.setFont(new Font("Tahoma", Font.BOLD, 14));
        lblLogin.setBackground(Color.WHITE);
        lblLogin.setBounds(158, 33, 120, 23);
        contentPane.add(lblLogin);

        lblUserName = new JLabel("Company User Name  ");
        lblUserName.setForeground(Color.BLACK);
        lblUserName.setBackground(Color.BLACK);
        lblUserName.setFont(new Font("Tahoma", Font.PLAIN, 14));
        lblUserName.setBounds(51, 103, 156, 23);
        contentPane.add(lblUserName);

        lblPwd = new JLabel("Password");
        lblPwd.setForeground(Color.BLACK);
        lblPwd.setFont(new Font("Tahoma", Font.PLAIN, 14));
        lblPwd.setBounds(50, 144, 121, 23);
        contentPane.add(lblPwd);

        txtUserName = new JTextField();
        txtUserName.setBounds(232, 106, 157, 20);
        contentPane.add(txtUserName);
        txtUserName.setColumns(10);

        btnLogin = new JButton("Login");
        btnLogin.addActionListener(new CompanyLoginView(this));
        btnLogin.setBackground(Color.WHITE);
        btnLogin.setFont(new Font("Tahoma", Font.PLAIN, 13));
        btnLogin.setBounds(158, 215, 106, 23);
        contentPane.add(btnLogin);

        txtPwd = new JPasswordField();
        txtPwd.setBounds(232, 147, 157, 20);
        contentPane.add(txtPwd);

        btnBack = new JButton("<--");
        btnBack.addActionListener(new CompanyLoginView(this));
        btnBack.setFont(new Font("Tahoma", Font.BOLD, 16));
        btnBack.setBounds(10, 11, 75, 23);
        contentPane.add(btnBack);

        lblBack = new JLabel("Back");
        lblBack.setHorizontalAlignment(SwingConstants.CENTER);
        lblBack.setBounds(10, 33, 75, 14);
        contentPane.add(lblBack);
    }

    public String getTxtCUserName() {
        return txtUserName.getText();
    }

    public String getPasswordField() {
        return txtPwd.getText();
    }
}